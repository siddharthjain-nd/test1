"""Detection and alignment correctness.

Alignment is the highest-risk code in the pipeline: a wrong transform degrades every
downstream number without ever raising an error. These tests check the geometry directly
rather than trusting it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faceindex import align, paths
from faceindex.detect import ScrfdDetector

DETECTOR_PATH = paths.models_dir() / "buffalo_sc" / "det_500m.onnx"


def test_umeyama_recovers_a_known_similarity_transform() -> None:
    """Rotate, scale and translate a point set; the solver must invert it exactly."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(5, 2)).astype(np.float32) * 30 + 100

    angle = np.radians(23.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    target = (2.5 * source @ rotation.T + np.array([17.0, -9.0])).astype(np.float32)

    matrix = align.umeyama_similarity(source, target)
    mapped = source @ matrix[:, :2].T + matrix[:, 2]

    np.testing.assert_allclose(mapped, target, atol=1e-3)


def test_umeyama_has_no_shear() -> None:
    """A similarity transform must preserve angles; affine would not."""
    rng = np.random.default_rng(1)
    source = rng.normal(size=(5, 2)).astype(np.float32) * 20 + 60
    target = rng.normal(size=(5, 2)).astype(np.float32) * 20 + 60

    linear = align.umeyama_similarity(source, target)[:, :2]
    # For scale*rotation, M @ M.T is a multiple of the identity.
    product = linear @ linear.T

    assert abs(product[0, 0] - product[1, 1]) < 1e-4
    assert abs(product[0, 1]) < 1e-4


def test_alignment_puts_landmarks_on_the_canonical_template() -> None:
    """The whole point of alignment: eyes land on the same pixels every time."""
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)

    # A plausible face: template scaled up 2x, rotated slightly, shifted into the image.
    angle = np.radians(10.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    landmarks = (align.ARCFACE_TEMPLATE * 2.0) @ rotation.T + np.array([90.0, 60.0])
    landmarks = landmarks.astype(np.float32)

    matrix = align.umeyama_similarity(landmarks, align.ARCFACE_TEMPLATE)
    mapped = landmarks @ matrix[:, :2].T + matrix[:, 2]

    np.testing.assert_allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-3)

    crop = align.align_face(image, landmarks)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8


def test_pose_estimation_signs() -> None:
    """Roll follows the eye line; yaw follows the nose offset along it."""
    frontal = align.ARCFACE_TEMPLATE.copy()
    yaw, roll = align.estimate_pose(frontal)
    assert abs(yaw) < 10.0
    assert abs(roll) < 1.0

    turned = frontal.copy()
    turned[2, 0] += 20.0  # nose pushed toward the right eye
    yaw_right, _ = align.estimate_pose(turned)
    assert yaw_right > yaw

    turned_left = frontal.copy()
    turned_left[2, 0] -= 20.0
    yaw_left, _ = align.estimate_pose(turned_left)
    assert yaw_left < yaw

    tilted = frontal.copy()
    tilted[1, 1] += 35.0  # right eye dropped
    _, roll_tilted = align.estimate_pose(tilted)
    assert roll_tilted > 5.0


def test_blur_attribute_separates_sharp_from_blurred() -> None:
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)

    import cv2

    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    landmarks = align.ARCFACE_TEMPLATE
    sharp_attrs = align.compute_attributes(sharp, landmarks, 400)
    blurred_attrs = align.compute_attributes(blurred, landmarks, 400)

    assert sharp_attrs.blur > blurred_attrs.blur * 5


def test_context_crop_handles_edge_faces() -> None:
    """A face at the image corner must still yield a valid crop."""
    image = np.full((200, 300, 3), 128, dtype=np.uint8)

    for bbox in (
        np.array([0.0, 0.0, 40.0, 40.0]),
        np.array([260.0, 160.0, 300.0, 200.0]),
        np.array([140.0, 90.0, 160.0, 110.0]),
    ):
        crop = align.context_crop(image, bbox)
        assert crop.shape == (256, 256, 3)


@pytest.mark.skipif(not DETECTOR_PATH.exists(), reason="detector not downloaded")
def test_detector_loads_and_returns_nothing_on_noise() -> None:
    """A noise image has no faces; anything returned would be a decoding bug."""
    detector = ScrfdDetector(DETECTOR_PATH, num_threads=2)
    rng = np.random.default_rng(4)
    noise = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    assert detector.detect(noise) == []
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


@pytest.mark.skipif(not DETECTOR_PATH.exists(), reason="detector not downloaded")
def test_detector_coordinates_are_in_image_space() -> None:
    """Whatever is detected must lie inside the image, at the original scale."""
    detector = ScrfdDetector(DETECTOR_PATH, score_threshold=0.3, num_threads=2)
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)

    for face in detector.detect(image):
        assert 0 <= face.bbox[0] < face.bbox[2] <= 1280
        assert 0 <= face.bbox[1] < face.bbox[3] <= 720
        assert face.landmarks.shape == (5, 2)


def test_landmarks_round_trip_through_json() -> None:
    """Landmarks are stored as JSON; the round trip must be lossless enough to realign."""
    landmarks = align.ARCFACE_TEMPLATE * 3.0 + 17.0
    restored = np.array(json.loads(json.dumps(landmarks.tolist())), dtype=np.float32)
    np.testing.assert_allclose(restored, landmarks, atol=1e-5)


def test_crop_directories_are_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACEINDEX_DATA", str(tmp_path))
    assert paths.crops_dir() == tmp_path / "crops"
    assert paths.context_crops_dir() == tmp_path / "context"
