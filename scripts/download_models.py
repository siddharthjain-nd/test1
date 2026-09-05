#!/usr/bin/env python3
"""Download and verify the ONNX models used by the pipeline.

LICENCE WARNING
    These weights come from the InsightFace model zoo and are licensed for
    NON-COMMERCIAL RESEARCH USE ONLY. They are downloaded here, never committed to
    this repository and never redistributed. See README.md.

Checksums are not hardcoded because upstream has re-cut archives in the past. Instead the
first successful run records SHA-256 digests into ``models/models.lock.json``; every later
run, including on the other machine, verifies against that lock. Commit the lock file so
both machines are provably running identical weights.

Usage
    python scripts/download_models.py                # fetch what is missing, verify the rest
    python scripts/download_models.py --packs buffalo_sc
    python scripts/download_models.py --verify-only  # no network access
    python scripts/download_models.py --force        # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceindex import paths

LOCK_FILENAME = "models.lock.json"
CHUNK_BYTES = 1 << 20
USER_AGENT = "faceindex-model-downloader/0.1"


@dataclass(frozen=True)
class ModelFile:
    """One ONNX file we extract from a pack."""

    name: str
    role: str
    note: str


@dataclass(frozen=True)
class ModelPack:
    """A zip archive published by InsightFace containing several ONNX models."""

    key: str
    url: str
    files: tuple[ModelFile, ...]
    description: str


# buffalo_sc is only ~16 MB and contains exactly the two models Phases 1-2 need.
# buffalo_l is ~326 MB and supplies the higher quality models used from Phase 5.
MODEL_PACKS: tuple[ModelPack, ...] = (
    ModelPack(
        key="buffalo_sc",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        description="Bootstrap pack for Phases 1-2 (small, fast).",
        files=(
            ModelFile("det_500m.onnx", "detector", "SCRFD-500M-KPS, bbox + 5 landmarks"),
            ModelFile("w600k_mbf.onnx", "embedder", "MobileFaceNet @ WebFace600K, 512-d"),
        ),
    ),
    ModelPack(
        key="buffalo_l",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        description="Quality pack for Phase 5 (large).",
        files=(
            ModelFile("det_10g.onnx", "detector", "SCRFD-10G-KPS, much better on small faces"),
            ModelFile("w600k_r50.onnx", "embedder", "ResNet50 @ WebFace600K, 512-d"),
        ),
    ),
)

LICENCE_NOTICE = """
------------------------------------------------------------------------------
  MODEL LICENCE NOTICE

  The weights about to be downloaded are from the InsightFace model zoo and are
  licensed for NON-COMMERCIAL RESEARCH USE ONLY.

  Their training datasets (WebFace600K, Glint360K, MS1M) carry further
  research-only restrictions.

  Do not use this project commercially. Do not redistribute these files.
------------------------------------------------------------------------------
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(lock_path: Path) -> dict[str, str]:
    if not lock_path.exists():
        return {}
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: {lock_path} is not valid JSON ({exc}). Delete it to rebuild."
        ) from exc
    checksums = data.get("sha256", {})
    if not isinstance(checksums, dict):
        raise SystemExit(f"error: {lock_path} has an unexpected structure. Delete it to rebuild.")
    return {str(k): str(v) for k, v in checksums.items()}


def save_lock(lock_path: Path, checksums: dict[str, str]) -> None:
    payload = {
        "_comment": (
            "SHA-256 of each extracted ONNX file. Commit this so every machine provably "
            "runs identical weights. Delete to re-record."
        ),
        "sha256": dict(sorted(checksums.items())),
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download(url: str, destination: Path) -> None:
    """Stream a URL to disk via a sibling .part file so a partial download never looks complete."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path = destination.with_name(destination.name + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as sink:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while chunk := response.read(CHUNK_BYTES):
                sink.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r    {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r    {downloaded / 1e6:7.1f} MB", end="", flush=True)
        print()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"error: download failed for {url}\n  {exc}") from exc

    # Same directory, so this is atomic and a crash can never leave a truncated model.
    tmp_path.replace(destination)


def extract_members(archive: Path, wanted: tuple[ModelFile, ...], target_dir: Path) -> list[Path]:
    """Pull the wanted ONNX files out of the archive, flattening any internal directory."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        members = {Path(info.filename).name: info for info in zf.infolist() if not info.is_dir()}
        for model_file in wanted:
            info = members.get(model_file.name)
            if info is None:
                available = ", ".join(sorted(members)) or "<empty>"
                raise SystemExit(
                    f"error: {model_file.name} not found in {archive.name}\n"
                    f"  archive contains: {available}"
                )
            destination = target_dir / model_file.name
            with zf.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink, CHUNK_BYTES)
            extracted.append(destination)
    return extracted


def process_pack(
    pack: ModelPack,
    models_root: Path,
    checksums: dict[str, str],
    *,
    force: bool,
    verify_only: bool,
) -> tuple[int, int]:
    """Ensure one pack is present and verified. Returns (ok_count, failure_count)."""
    target_dir = models_root / pack.key
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{pack.key}] {pack.description}")

    missing = [f for f in pack.files if not (target_dir / f.name).exists()]
    if (missing or force) and not verify_only:
        archive = models_root / f"{pack.key}.zip"
        if force or not archive.exists():
            print(f"  downloading {pack.url}")
            download(pack.url, archive)
        else:
            print(f"  reusing cached archive {archive.name}")
        print("  extracting")
        extract_members(archive, pack.files, target_dir)
        archive.unlink(missing_ok=True)
    elif missing and verify_only:
        for model_file in missing:
            print(f"  MISSING  {model_file.name}  (--verify-only, not downloading)")
        return 0, len(missing)

    ok = 0
    failed = 0
    for model_file in pack.files:
        path = target_dir / model_file.name
        key = f"{pack.key}/{model_file.name}"
        if not path.exists():
            print(f"  MISSING  {model_file.name}")
            failed += 1
            continue

        digest = sha256_of(path)
        size_mb = path.stat().st_size / 1e6
        recorded = checksums.get(key)
        if recorded is None:
            checksums[key] = digest
            state = "recorded"
        elif recorded == digest:
            state = "verified"
        else:
            print(
                f"  CHECKSUM MISMATCH  {model_file.name}\n"
                f"    expected {recorded}\n"
                f"    actual   {digest}\n"
                f"    The file differs from the locked version. Delete it and re-run, or "
                f"delete {LOCK_FILENAME} if upstream legitimately changed."
            )
            failed += 1
            continue

        print(f"  {state:8} {model_file.name:18} {size_mb:7.1f} MB  {model_file.role}")
        print(f"           {'':18} {'':7}     {model_file.note}")
        ok += 1

    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--packs",
        nargs="+",
        choices=[p.key for p in MODEL_PACKS],
        help="Only process these packs (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files without any network access.",
    )
    args = parser.parse_args()

    selected = [p for p in MODEL_PACKS if not args.packs or p.key in args.packs]

    models_root = paths.models_dir()
    models_root.mkdir(parents=True, exist_ok=True)
    lock_path = models_root / LOCK_FILENAME
    checksums = load_lock(lock_path)

    if not args.verify_only:
        print(LICENCE_NOTICE)

    total_ok = 0
    total_failed = 0
    for pack in selected:
        ok, failed = process_pack(
            pack, models_root, checksums, force=args.force, verify_only=args.verify_only
        )
        total_ok += ok
        total_failed += failed

    save_lock(lock_path, checksums)

    print(f"\n{total_ok} model(s) ok, {total_failed} problem(s).")
    print(f"Lock file: {lock_path}")
    if total_failed:
        print("Commit models.lock.json only once every model verifies.")
        return 1

    print("Commit models.lock.json so the other machine verifies against these digests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
