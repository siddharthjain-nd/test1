"""SCRFD face detection via ONNX Runtime.

Returns a bounding box, five facial landmarks and a confidence score per face. The
landmarks matter more than the box: alignment depends on them, and alignment is the single
largest source of silent accuracy loss in a face pipeline.

Model outputs are three strides (8, 16, 32), each with score, bbox-distance and
keypoint-distance heads, in that order -- nine tensors total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ArcFace preprocessing: (pixel - 127.5) / 128.0, RGB, NCHW. Not x/255.
_INPUT_MEAN = 127.5
_INPUT_SCALE = 1.0 / 128.0

_STRIDES = (8, 16, 32)
_ANCHORS_PER_CELL = 2


@dataclass(frozen=True)
class Face:
    """One detected face, in coordinates of the image that was passed in."""

    bbox: np.ndarray  # (4,) float32: x1, y1, x2, y2
    landmarks: np.ndarray  # (5, 2) float32: left eye, right eye, nose, mouth corners
    score: float

    @property
    def interocular_px(self) -> float:
        """Distance between the eye landmarks.

        A better size measure than box height: it is unaffected by how much hair, chin or
        background the detector chose to include.
        """
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def _anchor_centres(height: int, width: int, stride: int) -> np.ndarray:
    """Anchor centre coordinates for one stride, repeated per anchor."""
    ys, xs = np.mgrid[:height, :width]
    centres = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centres = centres.reshape(-1, 2)
    if _ANCHORS_PER_CELL > 1:
        centres = np.repeat(centres, _ANCHORS_PER_CELL, axis=0)
    return centres


def _distance_to_bbox(centres: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """SCRFD regresses distances from the anchor centre to each box edge."""
    x1 = centres[:, 0] - distances[:, 0]
    y1 = centres[:, 1] - distances[:, 1]
    x2 = centres[:, 0] + distances[:, 2]
    y2 = centres[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_landmarks(centres: np.ndarray, distances: np.ndarray) -> np.ndarray:
    points = [centres[:, i % 2] + distances[:, i] for i in range(distances.shape[1])]
    return np.stack(points, axis=-1).reshape(-1, 5, 2)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))

        xx1 = np.maximum(x1[best], x1[order[1:]])
        yy1 = np.maximum(y1[best], y1[order[1:]])
        xx2 = np.minimum(x2[best], x2[order[1:]])
        yy2 = np.minimum(y2[best], y2[order[1:]])

        overlap = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = overlap / (areas[best] + areas[order[1:]] - overlap)
        order = order[1:][iou <= threshold]

    return keep


class ScrfdDetector:
    """SCRFD detector. Not thread-safe; give each worker process its own instance."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        num_threads: int = 4,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        # CPU only: CoreML and CUDA do not agree bit-for-bit, and stored results must be
        # comparable across machines (PLAN.md section 3).
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        if len(self.session.get_outputs()) != 9:
            raise ValueError(
                f"expected 9 outputs (3 strides x score/bbox/kps), "
                f"got {len(self.session.get_outputs())}"
            )

    def detect(self, image_rgb: np.ndarray) -> list[Face]:
        """Detect faces in an RGB uint8 image. Coordinates are returned in its own scale."""
        height, width = image_rgb.shape[:2]
        if height == 0 or width == 0:
            return []

        target_w, target_h = self.input_size
        scale = min(target_w / width, target_h / height)
        resized_w, resized_h = round(width * scale), round(height * scale)

        # Letterbox at the top-left corner, which is what SCRFD was trained with.
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = cv2.resize(
            image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )

        blob = (canvas.astype(np.float32) - _INPUT_MEAN) * _INPUT_SCALE
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: blob})

        boxes_all: list[np.ndarray] = []
        landmarks_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []

        for index, stride in enumerate(_STRIDES):
            scores = outputs[index].reshape(-1)
            bbox_distances = outputs[index + 3].reshape(-1, 4) * stride
            kps_distances = outputs[index + 6].reshape(-1, 10) * stride

            keep = np.nonzero(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue

            centres = _anchor_centres(target_h // stride, target_w // stride, stride)
            boxes_all.append(_distance_to_bbox(centres[keep], bbox_distances[keep]))
            landmarks_all.append(_distance_to_landmarks(centres[keep], kps_distances[keep]))
            scores_all.append(scores[keep])

        if not boxes_all:
            return []

        boxes = np.vstack(boxes_all) / scale
        landmarks = np.vstack(landmarks_all) / scale
        scores = np.concatenate(scores_all)

        faces: list[Face] = []
        for index in _nms(boxes, scores, self.nms_threshold):
            box = boxes[index]
            # Clip to the image: SCRFD happily regresses past the edge for cropped faces.
            box = np.array(
                [
                    max(0.0, box[0]),
                    max(0.0, box[1]),
                    min(float(width), box[2]),
                    min(float(height), box[3]),
                ],
                dtype=np.float32,
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            faces.append(
                Face(
                    bbox=box,
                    landmarks=landmarks[index].astype(np.float32),
                    score=float(scores[index]),
                )
            )

        # Largest first, so "the main face in this photo" is faces[0].
        faces.sort(key=lambda f: f.width * f.height, reverse=True)
        return faces
