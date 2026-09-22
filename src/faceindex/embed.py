"""ArcFace embedding of aligned 112x112 crops.

Preprocessing is the single most dangerous thing in this file. ArcFace models are trained
on ``(pixel - 127.5) / 127.5`` over **RGB** channels in **NCHW** layout -- not ``x / 255``,
not BGR, not NHWC. Every one of those mistakes produces plausible-looking embeddings that
are silently wrong, and a silently wrong embedding poisons every downstream number without
raising a single error.

Nothing here validates itself. ``scripts/verify_embedding_parity.py`` is the only thing
that proves the preprocessing matches the reference implementation, and it must be run
once on each machine before any gold set is frozen.
"""

from __future__ import annotations

import platform
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from faceindex import facepool

# Bump when preprocessing, the model, or the normalisation changes in a way that makes
# stored vectors incomparable. Forces re-embedding rather than silently mixing versions.
EMBED_VERSION = "1"

EMBED_DIM = 512
CROP_SIZE = 112


@dataclass(frozen=True)
class EmbedConfig:
    """Everything that changes the numbers. Mirrors the ``embed`` block of a config file."""

    model_path: Path
    input_size: int = CROP_SIZE
    # InsightFace's reference ArcFaceONNX uses input_mean = input_std = 127.5.
    # CONTEXT.md section 6 states 128.0; the parity script is the arbiter, not either document.
    mean: float = 127.5
    std: float = 127.5
    flip_tta: bool = False
    num_threads: int = 4


class ArcFaceEmbedder:
    """ONNX ArcFace wrapper. Not thread-safe; give each worker its own instance."""

    def __init__(self, config: EmbedConfig) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.num_threads
        options.inter_op_num_threads = 1
        # CPU only. CoreML and CUDA do not agree bit-for-bit with it, and these vectors are
        # stored and compared across two machines (PLAN.md section 3).
        self.session = ort.InferenceSession(
            str(config.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.config = config
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            raise ValueError(f"expected a 4-D NCHW input, got shape {shape}")

    @property
    def model_name(self) -> str:
        return self.config.model_path.name

    def preprocess(self, crops_rgb: np.ndarray) -> np.ndarray:
        """``(N, 112, 112, 3)`` RGB uint8 -> ``(N, 3, 112, 112)`` float32, ArcFace-normalised."""
        if crops_rgb.ndim != 4 or crops_rgb.shape[-1] != 3:
            raise ValueError(f"expected (N, H, W, 3) RGB crops, got {crops_rgb.shape}")

        size = self.config.input_size
        if crops_rgb.shape[1] != size or crops_rgb.shape[2] != size:
            raise ValueError(f"expected {size}x{size} crops, got {crops_rgb.shape[1:3]}")

        batch = crops_rgb.astype(np.float32)
        batch = (batch - self.config.mean) / self.config.std
        # NHWC -> NCHW. Contiguous because ONNX Runtime copies non-contiguous input anyway.
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

    def _forward(self, crops_rgb: np.ndarray) -> np.ndarray:
        blob = self.preprocess(crops_rgb)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return np.asarray(outputs[0], dtype=np.float32)

    def embed(self, crops_rgb: np.ndarray) -> np.ndarray:
        """Embed a batch of aligned crops. Returns ``(N, 512)`` L2-normalised float32.

        With ``flip_tta`` the crop and its mirror are embedded, L2-normalised *separately*,
        averaged, and re-normalised. Averaging before normalising would weight whichever
        view happened to have the larger magnitude (Register C8).
        """
        vectors = self._forward(crops_rgb)

        if self.config.flip_tta:
            mirrored = self._forward(crops_rgb[:, :, ::-1, :])
            vectors = l2_normalise(vectors) + l2_normalise(mirrored)

        return l2_normalise(vectors)


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. Zero rows are left alone rather than producing NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms < 1e-12, 1.0, norms)


def to_blob(vector: np.ndarray) -> bytes:
    """Serialise one embedding for SQLite. Little-endian float32, fixed by contract."""
    return np.asarray(vector, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def load_matrix(blobs: list[bytes]) -> np.ndarray:
    """Stack stored embeddings into an ``(N, dim)`` matrix."""
    if not blobs:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.vstack([from_blob(b) for b in blobs]).astype(np.float32)


def runtime_fingerprint() -> tuple[str, str]:
    """``(platform, onnxruntime_version)`` -- stored per row so drift is detectable."""
    return f"{platform.system()}-{platform.machine()}", ort.__version__


# --------------------------------------------------------------------------------------
# The embedding pass over the face pool. Resumable at face granularity.
# --------------------------------------------------------------------------------------


def pending_faces(
    conn: sqlite3.Connection, model: str, *, limit: int | None = None
) -> list[tuple[int, str]]:
    """``(face_id, crop_path)`` for faces with no embedding *from this model*.

    Matching on the model, not only the version, is what makes switching models work.
    Keyed on version alone, every face already had an embedding from some model and the
    pass reported "nothing to do" while quietly refusing to run the new one.
    """
    sql = """
        SELECT f.id, f.crop_path FROM faces f
        LEFT JOIN face_embeddings e
               ON e.face_id = f.id AND e.embed_version = ? AND e.model = ?
        WHERE f.pool_version = ?
          AND f.crop_path IS NOT NULL
          AND e.face_id IS NULL
        ORDER BY f.id
    """
    params: list[object] = [EMBED_VERSION, model, facepool.POOL_VERSION]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [(int(r["id"]), str(r["crop_path"])) for r in conn.execute(sql, params)]


def load_crops(crop_paths: list[str]) -> tuple[np.ndarray, list[int]]:
    """Decode aligned crops. Returns ``(batch, ok_indices)``; unreadable crops are dropped.

    One unreadable crop must never abort a pass over 60k faces (CONTEXT.md rule 7).
    """
    images: list[np.ndarray] = []
    ok: list[int] = []
    for index, path in enumerate(crop_paths):
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                array = np.asarray(image, dtype=np.uint8)
        except Exception:
            continue
        if array.shape[:2] != (CROP_SIZE, CROP_SIZE):
            continue
        images.append(array)
        ok.append(index)

    if not images:
        return np.zeros((0, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8), []
    return np.stack(images), ok


def write_embeddings(
    conn: sqlite3.Connection,
    face_ids: list[int],
    vectors: np.ndarray,
    embedder: ArcFaceEmbedder,
) -> None:
    host, ort_version = runtime_fingerprint()
    now = datetime.now(UTC).isoformat()
    flip = 1 if embedder.config.flip_tta else 0

    conn.executemany(
        """
        INSERT INTO face_embeddings (
            face_id, model, embedding, dim, embed_version,
            platform, onnxruntime_version, flip_tta, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(face_id, model) DO UPDATE SET
            embedding=excluded.embedding, dim=excluded.dim,
            embed_version=excluded.embed_version, platform=excluded.platform,
            onnxruntime_version=excluded.onnxruntime_version,
            flip_tta=excluded.flip_tta, created_at=excluded.created_at
        """,
        [
            (
                face_id,
                embedder.model_name,
                to_blob(vector),
                int(vector.shape[0]),
                EMBED_VERSION,
                host,
                ort_version,
                flip,
                now,
            )
            for face_id, vector in zip(face_ids, vectors, strict=True)
        ],
    )


def run(
    conn: sqlite3.Connection,
    tasks: list[tuple[int, str]],
    embedder: ArcFaceEmbedder,
    *,
    batch_size: int = 64,
    commit_every: int = 2048,
) -> Iterator[tuple[int, int]]:
    """Embed pending faces, yielding ``(done, skipped)`` counts per batch."""
    since_commit = 0

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        crops, ok = load_crops([path for _, path in chunk])
        skipped = len(chunk) - len(ok)

        if ok:
            vectors = embedder.embed(crops)
            write_embeddings(conn, [chunk[i][0] for i in ok], vectors, embedder)
            since_commit += len(ok)

        if since_commit >= commit_every:
            conn.commit()
            since_commit = 0

        yield len(ok), skipped

    conn.commit()
