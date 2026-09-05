"""Face alignment and per-face attribute extraction.

Alignment warps the five detected landmarks onto ArcFace's canonical 112x112 positions.
Every ArcFace-family model was trained on faces warped this way; feeding a raw crop instead
silently degrades every downstream number, so this module is worth reading carefully.

Attributes are computed here rather than labelled by hand. They exist so errors can be
sliced -- "F1 is 0.92 overall but 0.61 on profile faces" is actionable, a single aggregate
number is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ArcFace canonical landmark positions for a 112x112 crop, in
# (left eye, right eye, nose, left mouth corner, right mouth corner) order.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

CROP_SIZE = 112


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation).

    Implements Umeyama (1991). A similarity transform is used rather than a full affine
    because affine allows shear, which distorts face geometry and shifts the embedding.

    Returns a 2x3 matrix suitable for ``cv2.warpAffine``.
    """
    num_points, dimensions = source.shape

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centred = source - source_mean
    target_centred = target - target_mean

    covariance = target_centred.T @ source_centred / num_points
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.ones(dimensions)
    if np.linalg.det(covariance) < 0:
        correction[-1] = -1.0

    rank = np.linalg.matrix_rank(covariance)
    if rank == dimensions - 1 and np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1] = -1.0

    rotation = u @ np.diag(correction) @ vt

    source_variance = source_centred.var(axis=0).sum()
    scale = 1.0 if source_variance == 0 else (singular_values * correction).sum() / source_variance

    translation = target_mean - scale * rotation @ source_mean

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def align_face(image_rgb: np.ndarray, landmarks: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Warp a face onto the canonical template. Returns an RGB uint8 crop."""
    template = ARCFACE_TEMPLATE if size == CROP_SIZE else ARCFACE_TEMPLATE * (size / CROP_SIZE)
    matrix = umeyama_similarity(landmarks.astype(np.float32), template)
    return cv2.warpAffine(
        image_rgb, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )


def context_crop(
    image_rgb: np.ndarray, bbox: np.ndarray, *, scale: float = 2.0, size: int = 256
) -> np.ndarray:
    """A wider, unwarped crop for human labelling only.

    A bare aligned 112x112 face is often too tight for a person to judge identity; the
    surrounding context is what makes labelling fast. Never fed to a model.
    """
    height, width = image_rgb.shape[:2]
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale / 2.0

    x1 = int(max(0, round(cx - half)))
    y1 = int(max(0, round(cy - half)))
    x2 = int(min(width, round(cx + half)))
    y2 = int(min(height, round(cy + half)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)

    return cv2.resize(image_rgb[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class FaceAttributes:
    """Auto-derived per-face metadata. Never hand-labelled."""

    interocular_px: float
    relative_size: float  # interocular distance as a fraction of image height
    yaw_deg: float
    roll_deg: float
    blur: float
    brightness: float
    dark_fraction: float
    bright_fraction: float


def estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Rough yaw and roll in degrees from the five landmarks.

    Not a 3D pose estimate: yaw is derived from how far the nose sits from the midpoint
    between the eyes, normalised by interocular distance. Good enough to bucket faces into
    frontal / semi-profile / profile, which is all the slicing needs.
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]

    delta = right_eye - left_eye
    roll = float(np.degrees(np.arctan2(delta[1], delta[0])))

    interocular = float(np.linalg.norm(delta))
    if interocular < 1e-6:
        return 0.0, roll

    eye_centre = (left_eye + right_eye) / 2.0
    # Project the nose offset onto the eye axis so head tilt does not leak into yaw.
    axis = delta / interocular
    offset = float(np.dot(nose - eye_centre, axis)) / interocular

    yaw = float(np.clip(offset * 180.0, -90.0, 90.0))
    return yaw, roll


def compute_attributes(
    aligned_rgb: np.ndarray, landmarks: np.ndarray, image_height: int
) -> FaceAttributes:
    """Derive size, pose and image-quality attributes for one face."""
    grey = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)

    interocular = float(np.linalg.norm(landmarks[1] - landmarks[0]))
    yaw, roll = estimate_pose(landmarks)

    # Laplacian variance is the standard sharpness proxy. Measured on the aligned crop so
    # it is already normalised for face size -- a small sharp face and a large blurry one
    # would otherwise score alike.
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    return FaceAttributes(
        interocular_px=interocular,
        relative_size=interocular / image_height if image_height else 0.0,
        yaw_deg=yaw,
        roll_deg=roll,
        blur=blur,
        brightness=float(grey.mean()),
        dark_fraction=float((grey < 32).mean()),
        bright_fraction=float((grey > 224).mean()),
    )
