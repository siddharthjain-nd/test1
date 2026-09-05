"""Face pool construction: decode, detect, align, and store crops plus attributes.

This is the expensive pass and the only one that must read the original photos. Everything
downstream (embedding, clustering, threshold tuning, the whole experiment loop) works from
the 112x112 crops this produces, so it is worth doing well once.

Resumable at photo granularity: interrupt it and rerun the same command.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from faceindex import align
from faceindex.detect import Face, ScrfdDetector
from faceindex.ingest import FACE_CANDIDATE_KINDS

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = 400_000_000

# Bump when detection, alignment or attribute extraction changes in a way that
# invalidates stored faces. Forces reprocessing.
POOL_VERSION = "1"

_worker_detector: ScrfdDetector | None = None
_worker_config: PoolConfig | None = None


@dataclass(frozen=True)
class PoolConfig:
    detector_path: Path
    crops_dir: Path
    context_dir: Path
    max_long_edge: int = 2048
    det_input_size: int = 640
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    context_scale: float = 2.0
    context_size: int = 256
    threads_per_worker: int = 2
    save_context: bool = True


@dataclass
class PhotoResult:
    photo_id: int
    faces: list[dict[str, object]]
    error: str | None = None


def decode(path: Path, max_long_edge: int) -> tuple[np.ndarray, float]:
    """Decode an image at bounded size. Returns ``(rgb_array, scale_to_original)``.

    ``Image.draft`` performs JPEG DCT-domain downscaling, which is typically 4-8x faster
    than decoding at full size and then resizing. It is the single largest speed win in
    this pass.
    """
    with Image.open(path) as image:
        original_long_edge = max(image.size)
        image.draft("RGB", (max_long_edge, max_long_edge))

        # Rotate before detection, or most portrait photos yield no faces.
        oriented = ImageOps.exif_transpose(image) or image
        if oriented.mode != "RGB":
            oriented = oriented.convert("RGB")

        if max(oriented.size) > max_long_edge:
            ratio = max_long_edge / max(oriented.size)
            oriented = oriented.resize(
                (round(oriented.width * ratio), round(oriented.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        array = np.asarray(oriented, dtype=np.uint8)

    decoded_long_edge = max(array.shape[:2])
    scale = original_long_edge / decoded_long_edge if decoded_long_edge else 1.0
    return array, scale


def crop_paths(config: PoolConfig, photo_id: int, face_index: int) -> tuple[Path, Path]:
    """Shard crops across subdirectories; tens of thousands of files in one folder is slow."""
    shard = f"{photo_id % 100:02d}"
    name = f"{photo_id}_{face_index}.jpg"
    return config.crops_dir / shard / name, config.context_dir / shard / name


def _init_worker(config: PoolConfig) -> None:
    global _worker_detector, _worker_config
    # Each worker owns its session; oversubscribing BLAS threads inside a process pool
    # causes thrashing rather than speed.
    os.environ.setdefault("OMP_NUM_THREADS", str(config.threads_per_worker))
    _worker_config = config
    _worker_detector = ScrfdDetector(
        config.detector_path,
        input_size=(config.det_input_size, config.det_input_size),
        score_threshold=config.score_threshold,
        nms_threshold=config.nms_threshold,
        num_threads=config.threads_per_worker,
    )


def _face_row(
    face: Face, aligned: np.ndarray, image_height: int, scale: float, index: int
) -> dict[str, object]:
    attributes = align.compute_attributes(aligned, face.landmarks, image_height)
    return {
        "face_index": index,
        "bbox_x1": float(face.bbox[0]),
        "bbox_y1": float(face.bbox[1]),
        "bbox_x2": float(face.bbox[2]),
        "bbox_y2": float(face.bbox[3]),
        "landmarks": json.dumps(face.landmarks.tolist()),
        "det_score": face.score,
        "interocular_px": attributes.interocular_px,
        "relative_size": attributes.relative_size,
        "yaw_deg": attributes.yaw_deg,
        "roll_deg": attributes.roll_deg,
        "blur": attributes.blur,
        "brightness": attributes.brightness,
        "dark_fraction": attributes.dark_fraction,
        "bright_fraction": attributes.bright_fraction,
        "decode_scale": scale,
    }


def process_photo(task: tuple[int, str]) -> PhotoResult:
    """Worker entry point: one photo in, its faces out."""
    photo_id, path_str = task
    assert _worker_detector is not None and _worker_config is not None
    config = _worker_config

    try:
        image, scale = decode(Path(path_str), config.max_long_edge)
    except Exception as exc:
        return PhotoResult(photo_id, [], f"decode: {type(exc).__name__}: {exc}")

    try:
        detections = _worker_detector.detect(image)
    except Exception as exc:
        return PhotoResult(photo_id, [], f"detect: {type(exc).__name__}: {exc}")

    rows: list[dict[str, object]] = []
    for index, face in enumerate(detections):
        try:
            aligned = align.align_face(image, face.landmarks)
            crop_path, context_path = crop_paths(config, photo_id, index)
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(aligned).save(crop_path, "JPEG", quality=95)

            if config.save_context:
                context_path.parent.mkdir(parents=True, exist_ok=True)
                context = align.context_crop(
                    image, face.bbox, scale=config.context_scale, size=config.context_size
                )
                Image.fromarray(context).save(context_path, "JPEG", quality=85)

            row = _face_row(face, aligned, image.shape[0], scale, index)
            row["crop_path"] = str(crop_path)
            row["context_path"] = str(context_path) if config.save_context else None
            rows.append(row)
        except Exception as exc:
            return PhotoResult(photo_id, rows, f"align[{index}]: {type(exc).__name__}: {exc}")

    return PhotoResult(photo_id, rows)


def pending_photos(conn: sqlite3.Connection, *, limit: int | None = None) -> list[tuple[int, str]]:
    """Face candidates not yet processed at the current pool version."""
    placeholders = ",".join("?" for _ in FACE_CANDIDATE_KINDS)
    sql = f"""
        SELECT p.id, p.path FROM photos p
        LEFT JOIN photo_pool_status s ON s.photo_id = p.id AND s.pool_version = ?
        WHERE p.kind IN ({placeholders})
          AND p.duplicate_of IS NULL
          AND s.photo_id IS NULL
        ORDER BY p.id
    """
    params: list[object] = [POOL_VERSION, *sorted(FACE_CANDIDATE_KINDS)]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [(int(r["id"]), str(r["path"])) for r in conn.execute(sql, params)]


def write_result(conn: sqlite3.Connection, result: PhotoResult, detector_name: str) -> None:
    now = datetime.now(UTC).isoformat()

    conn.execute(
        "DELETE FROM faces WHERE photo_id = ? AND pool_version = ?", (result.photo_id, POOL_VERSION)
    )
    for row in result.faces:
        conn.execute(
            """
            INSERT INTO faces (
                photo_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2, landmarks,
                det_score, interocular_px, relative_size, yaw_deg, roll_deg, blur,
                brightness, dark_fraction, bright_fraction, decode_scale,
                crop_path, context_path, detector, pool_version, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.photo_id,
                row["face_index"],
                row["bbox_x1"],
                row["bbox_y1"],
                row["bbox_x2"],
                row["bbox_y2"],
                row["landmarks"],
                row["det_score"],
                row["interocular_px"],
                row["relative_size"],
                row["yaw_deg"],
                row["roll_deg"],
                row["blur"],
                row["brightness"],
                row["dark_fraction"],
                row["bright_fraction"],
                row["decode_scale"],
                row["crop_path"],
                row["context_path"],
                detector_name,
                POOL_VERSION,
                now,
            ),
        )

    conn.execute(
        "INSERT INTO photo_pool_status (photo_id, pool_version, n_faces, error, processed_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(photo_id) DO UPDATE SET "
        "pool_version=excluded.pool_version, n_faces=excluded.n_faces, "
        "error=excluded.error, processed_at=excluded.processed_at",
        (result.photo_id, POOL_VERSION, len(result.faces), result.error, now),
    )


def run(
    conn: sqlite3.Connection,
    tasks: list[tuple[int, str]],
    config: PoolConfig,
    *,
    workers: int,
    commit_every: int = 50,
) -> Iterator[PhotoResult]:
    """Process photos in parallel, yielding results as they complete."""
    detector_name = config.detector_path.name
    pending = 0

    if workers <= 1:
        _init_worker(config)
        for task in tasks:
            result = process_photo(task)
            write_result(conn, result, detector_name)
            pending += 1
            if pending >= commit_every:
                conn.commit()
                pending = 0
            yield result
    else:
        context = mp.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
            for result in pool.imap_unordered(process_photo, tasks, chunksize=8):
                write_result(conn, result, detector_name)
                pending += 1
                if pending >= commit_every:
                    conn.commit()
                    pending = 0
                yield result

    conn.commit()
