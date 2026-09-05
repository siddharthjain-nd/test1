"""Corpus scanning: walk, classify, hash, dedup, extract EXIF.

Deliberately cheap. Nothing here decodes pixel data -- ``Image.open`` only parses the
header, so a 134 GB library is bounded by directory traversal and ~128 KB read per
candidate file rather than by decode cost. Videos are rejected on extension alone and
are never opened.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pillow_heif
from PIL import ExifTags, Image, UnidentifiedImageError

from faceindex import store

pillow_heif.register_heif_opener()

# Guard against decompression-bomb DoS while still allowing large camera files.
Image.MAX_IMAGE_PIXELS = 400_000_000

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}
)

# Rejected on extension alone, before any I/O. Opening these would read gigabytes.
VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".3gp",
        ".3gpp",
        ".mpg",
        ".mpeg",
        ".mpe",
        ".wmv",
        ".flv",
        ".webm",
        ".mts",
        ".m2ts",
        ".ogv",
        ".asf",
        ".rm",
        ".rmvb",
    }
)

AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".amr"})

# Flagged rather than silently ignored: an archive in a photo library usually contains photos.
ARCHIVE_EXTENSIONS = frozenset({".zip", ".rar", ".7z", ".tar", ".gz"})

# Camera RAW. Out of scope for now; recorded so the count is visible rather than silent.
RAW_EXTENSIONS = frozenset({".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"})

_SCREENSHOT_NAME = re.compile(r"screen[\s_-]?shot|screenshot", re.IGNORECASE)
# WhatsApp: IMG-20230115-WA0001.jpg / VID-...; Telegram: photo_2023-01-15_....jpg
_FORWARDED_NAME = re.compile(r"-WA\d{4}|^photo_\d{4}-\d{2}-\d{2}|^IMG-\d{8}-WA", re.IGNORECASE)

# Messaging apps strip EXIF but keep the date in the filename, which is where most of a
# typical library's undated photos get their dates back from.
# Matches IMG-20230115-WA0001, IMG_20230115_143022, PXL_20230115_..., 2023-01-15 14.30.22
_DATE_IN_NAME = re.compile(
    r"(?<!\d)(?P<y>19\d{2}|20\d{2})(?P<sep>[-_.]?)(?P<m>0[1-9]|1[0-2])(?P=sep)"
    r"(?P<d>0[1-9]|[12]\d|3[01])(?!\d)"
)
_TIME_IN_NAME = re.compile(
    # Trailing \d{0,3} absorbs the milliseconds Pixel appends: PXL_20220704_101530123.jpg
    r"(?<!\d)(?P<H>[01]\d|2[0-3])(?P<sep>[-_.:]?)(?P<M>[0-5]\d)(?P=sep)(?P<S>[0-5]\d)\d{0,3}(?!\d)"
)
_YEAR_IN_FOLDER = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")

EARLIEST_PLAUSIBLE_YEAR = 1990

_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME_DIGITIZED = 36868
_EXIF_DATETIME = 306
_EXIF_MAKE = 271
_EXIF_MODEL = 272
_EXIF_ORIENTATION = 274

QUICK_HASH_BYTES = 65536

# Anything smaller cannot contain a usable face; icons, spacers, thumbnails.
MIN_PIXELS = 10_000


class Kind:
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    FORWARDED = "forwarded"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    RAW = "raw"
    TINY = "tiny"
    UNSUPPORTED = "unsupported"
    UNREADABLE = "unreadable"


#: Kinds that proceed to face detection.
FACE_CANDIDATE_KINDS = frozenset({Kind.PHOTO, Kind.FORWARDED})


@dataclass
class PhotoRecord:
    path: Path
    rel_path: str
    size_bytes: int
    mtime: float
    kind: str
    reason: str | None = None
    quick_hash: str | None = None
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    taken_at: str | None = None
    taken_at_source: str | None = None
    exif_modified_at: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    orientation: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None


@dataclass
class ScanStats:
    seen: int = 0
    processed: int = 0
    skipped_resumed: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record(self, kind: str) -> None:
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


class DateSource:
    EXIF_ORIGINAL = "exif_original"
    EXIF_DIGITIZED = "exif_digitized"
    FILENAME = "filename"
    EXIF_MODIFIED = "exif_modified"
    FOLDER = "folder"
    MTIME = "mtime"


def _plausible(year: int) -> bool:
    return EARLIEST_PLAUSIBLE_YEAR <= year <= datetime.now(UTC).year + 1


def date_from_filename(name: str) -> str | None:
    """Recover a capture date from a camera or messaging-app filename.

    Returns an ISO-8601 string, or None when the name carries no plausible date.
    """
    match = _DATE_IN_NAME.search(name)
    if match is None:
        return None

    year, month, day = int(match["y"]), int(match["m"]), int(match["d"])
    if not _plausible(year):
        return None

    hour = minute = second = 0
    time_match = _TIME_IN_NAME.search(name, match.end())
    if time_match is not None:
        hour, minute, second = (
            int(time_match["H"]),
            int(time_match["M"]),
            int(time_match["S"]),
        )

    try:
        return datetime(year, month, day, hour, minute, second).isoformat()
    except ValueError:
        return None


def year_from_folder(rel_path: str) -> str | None:
    """Fall back to a year named in a parent folder, e.g. ``2016 Goa/``.

    Only a year is claimed, so the result is deliberately coarse: January 1st.
    """
    for part in reversed(Path(rel_path).parts[:-1]):
        match = _YEAR_IN_FOLDER.search(part)
        if match and _plausible(int(match[1])):
            return datetime(int(match[1]), 1, 1).isoformat()
    return None


def backfill_dates(conn: sqlite3.Connection, *, use_mtime: bool = False) -> dict[str, int]:
    """Fill missing ``taken_at`` from weaker sources, in descending order of trust.

    Priority: EXIF DateTimeOriginal/Digitized (set during scanning) > filename >
    EXIF DateTime (tag 306) > folder year > mtime.

    **Filename beats EXIF tag 306 deliberately.** Tag 306 is a modification time: bulk
    edits, exports and copies rewrite it, which shows up as dozens of unrelated photos
    sharing a timestamp to the second. A camera or messaging filename is the more
    reliable signal.

    Database-only: reads no files, so this is fast and safe to rerun. Capture-tag EXIF
    dates are never overwritten.

    ``use_mtime`` is off by default because copying a library resets modification times,
    and a wrong date is worse than no date -- it would feed a false signal into the
    Phase 4 time prior.
    """
    counts = {
        DateSource.FILENAME: 0,
        DateSource.EXIF_MODIFIED: 0,
        DateSource.FOLDER: 0,
        DateSource.MTIME: 0,
    }
    rows = conn.execute(
        "SELECT id, rel_path, mtime, exif_modified_at FROM photos "
        "WHERE taken_at IS NULL AND kind NOT IN (?, ?, ?, ?, ?)",
        (Kind.UNREADABLE, Kind.UNSUPPORTED, Kind.VIDEO, Kind.AUDIO, Kind.ARCHIVE),
    ).fetchall()

    for row in rows:
        rel_path = str(row["rel_path"])
        resolved = date_from_filename(Path(rel_path).name)
        source = DateSource.FILENAME

        if resolved is None and row["exif_modified_at"]:
            resolved = str(row["exif_modified_at"])
            source = DateSource.EXIF_MODIFIED

        if resolved is None:
            resolved = year_from_folder(rel_path)
            source = DateSource.FOLDER

        if resolved is None and use_mtime and row["mtime"]:
            candidate = datetime.fromtimestamp(float(row["mtime"]), tz=UTC)
            if _plausible(candidate.year):
                resolved = candidate.replace(tzinfo=None).isoformat()
                source = DateSource.MTIME

        if resolved is None:
            continue

        conn.execute(
            "UPDATE photos SET taken_at = ?, taken_at_source = ? WHERE id = ?",
            (resolved, source, row["id"]),
        )
        counts[source] += 1

    conn.commit()
    return counts


def filename_exif_disagreements(conn: sqlite3.Connection, *, days: int = 7) -> tuple[int, int]:
    """Count photos whose filename date and EXIF capture date disagree by more than ``days``.

    A high rate means one of the two sources is untrustworthy for this corpus and the
    priority order deserves revisiting. Returns ``(disagreements, comparable_rows)``.
    """
    rows = conn.execute(
        "SELECT rel_path, taken_at FROM photos WHERE taken_at IS NOT NULL "
        "AND taken_at_source IN (?, ?)",
        (DateSource.EXIF_ORIGINAL, DateSource.EXIF_DIGITIZED),
    ).fetchall()

    comparable = 0
    disagreements = 0
    for row in rows:
        from_name = date_from_filename(Path(str(row["rel_path"])).name)
        if from_name is None:
            continue
        comparable += 1
        delta = datetime.fromisoformat(str(row["taken_at"])) - datetime.fromisoformat(from_name)
        if abs(delta.days) > days:
            disagreements += 1

    return disagreements, comparable


def iter_files(root: Path, *, follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield every regular file under ``root``, skipping hidden and system directories."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d not in {"@eaDir", "__MACOSX"}
        ]
        for filename in filenames:
            if filename.startswith("."):
                continue
            yield Path(dirpath) / filename


def quick_hash(path: Path, size_bytes: int, *, chunk: int = QUICK_HASH_BYTES) -> str:
    """Digest of (size, first chunk, last chunk).

    Avoids reading whole files. Collisions are possible, so callers must confirm any
    apparent duplicate with a full SHA-256 before discarding data.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(size_bytes).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size_bytes > 2 * chunk:
            handle.seek(-chunk, os.SEEK_END)
            digest.update(handle.read(chunk))
    return digest.hexdigest()


def content_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _parse_exif_datetime(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return None


def _rational_to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dms_to_degrees(dms: Any, ref: Any) -> float | None:
    """Convert EXIF degrees/minutes/seconds rationals to a signed decimal degree."""
    try:
        degrees, minutes, seconds = (_rational_to_float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    if degrees is None or minutes is None or seconds is None:
        return None

    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if isinstance(ref, str) and ref.upper() in {"S", "W"}:
        decimal = -decimal
    return decimal if -180.0 <= decimal <= 180.0 else None


def _extract_gps(exif: Any) -> tuple[float | None, float | None]:
    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (AttributeError, KeyError, OSError, ValueError):
        return None, None
    if not gps:
        return None, None

    lat = _dms_to_degrees(gps.get(2), gps.get(1))
    lon = _dms_to_degrees(gps.get(4), gps.get(3))
    return lat, lon


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("\x00").strip()
    return text or None


def read_header(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read dimensions, format and EXIF without decoding pixels.

    Returns ``(metadata, error)``. ``Image.open`` is lazy: it parses the header only.
    """
    metadata: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            metadata["width"], metadata["height"] = image.size
            metadata["image_format"] = image.format

            try:
                exif = image.getexif()
            except (OSError, ValueError, SyntaxError):
                exif = None

            if exif:
                metadata["camera_make"] = _clean(exif.get(_EXIF_MAKE))
                metadata["camera_model"] = _clean(exif.get(_EXIF_MODEL))

                orientation = exif.get(_EXIF_ORIENTATION)
                if isinstance(orientation, int) and 1 <= orientation <= 8:
                    metadata["orientation"] = orientation

                taken = None
                taken_source = None
                try:
                    sub_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                except (AttributeError, KeyError, OSError, ValueError):
                    sub_ifd = {}

                for tag, source in (
                    (_EXIF_DATETIME_ORIGINAL, DateSource.EXIF_ORIGINAL),
                    (_EXIF_DATETIME_DIGITIZED, DateSource.EXIF_DIGITIZED),
                ):
                    taken = _parse_exif_datetime(sub_ifd.get(tag))
                    if taken:
                        taken_source = source
                        break

                metadata["taken_at"] = taken
                metadata["taken_at_source"] = taken_source
                # Tag 306 is a modification time. Kept apart so it can rank below filenames.
                metadata["exif_modified_at"] = _parse_exif_datetime(exif.get(_EXIF_DATETIME))

                lat, lon = _extract_gps(exif)
                metadata["gps_lat"], metadata["gps_lon"] = lat, lon
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return metadata, f"{type(exc).__name__}: {exc}"

    return metadata, None


def classify(path: Path, metadata: dict[str, Any]) -> tuple[str, str | None]:
    """Decide what a file is, given its name and header metadata.

    Documents, receipts and memes are deliberately *not* classified here -- doing so
    reliably needs a model. They fall through as photos and are harmless: they simply
    yield no faces.
    """
    name = path.name

    if _FORWARDED_NAME.search(name) or "whatsapp" in str(path.parent).lower():
        return Kind.FORWARDED, "messaging-app filename or folder"

    if _SCREENSHOT_NAME.search(name):
        return Kind.SCREENSHOT, "filename"

    width = metadata.get("width") or 0
    height = metadata.get("height") or 0
    if width * height < MIN_PIXELS:
        return Kind.TINY, f"{width}x{height} below {MIN_PIXELS}px"

    has_camera = bool(metadata.get("camera_make") or metadata.get("camera_model"))
    if metadata.get("image_format") == "PNG" and not has_camera:
        return Kind.SCREENSHOT, "PNG without camera EXIF"

    return Kind.PHOTO, None


def build_record(path: Path, root: Path, *, compute_hash: bool = True) -> PhotoRecord:
    """Produce one row's worth of information for a single file."""
    try:
        stat = path.stat()
    except OSError as exc:
        return PhotoRecord(
            path=path,
            rel_path=path.name,
            size_bytes=0,
            mtime=0.0,
            kind=Kind.UNREADABLE,
            reason=f"stat failed: {exc}",
        )

    try:
        rel_path = str(path.relative_to(root))
    except ValueError:
        rel_path = str(path)

    size_bytes = stat.st_size
    mtime = stat.st_mtime

    def rejected(kind: str, reason: str) -> PhotoRecord:
        return PhotoRecord(
            path=path,
            rel_path=rel_path,
            size_bytes=size_bytes,
            mtime=mtime,
            kind=kind,
            reason=reason,
        )

    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return rejected(Kind.VIDEO, "video extension")
    if suffix in AUDIO_EXTENSIONS:
        return rejected(Kind.AUDIO, "audio extension")
    if suffix in ARCHIVE_EXTENSIONS:
        return rejected(Kind.ARCHIVE, "archive extension")
    if suffix in RAW_EXTENSIONS:
        return rejected(Kind.RAW, "raw extension")
    if suffix not in IMAGE_EXTENSIONS:
        return rejected(Kind.UNSUPPORTED, f"extension {suffix or '<none>'}")
    if size_bytes == 0:
        return rejected(Kind.UNREADABLE, "zero bytes")

    metadata, error = read_header(path)
    if error is not None:
        return rejected(Kind.UNREADABLE, error)

    kind, reason = classify(path, metadata)

    return PhotoRecord(
        path=path,
        rel_path=rel_path,
        size_bytes=size_bytes,
        mtime=mtime,
        kind=kind,
        reason=reason,
        quick_hash=quick_hash(path, size_bytes) if compute_hash else None,
        width=metadata.get("width"),
        height=metadata.get("height"),
        image_format=metadata.get("image_format"),
        taken_at=metadata.get("taken_at"),
        taken_at_source=metadata.get("taken_at_source"),
        exif_modified_at=metadata.get("exif_modified_at"),
        camera_make=metadata.get("camera_make"),
        camera_model=metadata.get("camera_model"),
        orientation=metadata.get("orientation"),
        gps_lat=metadata.get("gps_lat"),
        gps_lon=metadata.get("gps_lon"),
    )


def _insert(conn: sqlite3.Connection, record: PhotoRecord, scanned_at: str) -> None:
    conn.execute(
        """
        INSERT INTO photos (
            path, rel_path, size_bytes, mtime, quick_hash, kind, reason,
            width, height, image_format, taken_at, taken_at_source, exif_modified_at,
            camera_make, camera_model, orientation, gps_lat, gps_lon, scanned_at, scan_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            rel_path=excluded.rel_path, size_bytes=excluded.size_bytes,
            mtime=excluded.mtime, quick_hash=excluded.quick_hash, kind=excluded.kind,
            reason=excluded.reason, width=excluded.width, height=excluded.height,
            image_format=excluded.image_format, taken_at=excluded.taken_at,
            taken_at_source=excluded.taken_at_source,
            exif_modified_at=excluded.exif_modified_at,
            camera_make=excluded.camera_make, camera_model=excluded.camera_model,
            orientation=excluded.orientation, gps_lat=excluded.gps_lat,
            gps_lon=excluded.gps_lon, scanned_at=excluded.scanned_at,
            scan_version=excluded.scan_version, content_hash=NULL, duplicate_of=NULL
        """,
        (
            str(record.path),
            record.rel_path,
            record.size_bytes,
            record.mtime,
            record.quick_hash,
            record.kind,
            record.reason,
            record.width,
            record.height,
            record.image_format,
            record.taken_at,
            record.taken_at_source,
            record.exif_modified_at,
            record.camera_make,
            record.camera_model,
            record.orientation,
            record.gps_lat,
            record.gps_lon,
            scanned_at,
            store.SCAN_VERSION,
        ),
    )


def scan(
    conn: sqlite3.Connection,
    root: Path,
    *,
    follow_symlinks: bool = False,
    commit_every: int = 500,
    progress: Iterable[Path] | None = None,
) -> ScanStats:
    """Walk ``root`` and record every file. Resumable and stream-only.

    ``progress`` optionally wraps the file iterator (e.g. with tqdm).
    """
    stats = ScanStats()
    seen_before = store.already_scanned(conn)
    scanned_at = datetime.now(UTC).isoformat()

    files = iter_files(root, follow_symlinks=follow_symlinks) if progress is None else progress
    pending = 0

    for path in files:
        stats.seen += 1

        previous = seen_before.get(str(path))
        if previous is not None:
            try:
                stat = path.stat()
            except OSError:
                stat = None
            if stat is not None and (stat.st_size, stat.st_mtime) == previous:
                stats.skipped_resumed += 1
                continue

        try:
            record = build_record(path, root)
        except Exception as exc:  # a single bad file must never abort a multi-hour scan
            stats.errors.append(f"{path}: {type(exc).__name__}: {exc}")
            stats.record(Kind.UNREADABLE)
            continue

        _insert(conn, record, scanned_at)
        stats.processed += 1
        stats.record(record.kind)

        pending += 1
        if pending >= commit_every:
            conn.commit()
            pending = 0

    conn.commit()
    return stats


def resolve_duplicates(conn: sqlite3.Connection) -> tuple[int, int]:
    """Confirm quick-hash collisions with SHA-256 and mark non-canonical copies.

    The canonical copy is the one with the oldest mtime; ties break on path so the result
    is deterministic. Returns ``(duplicate_count, bytes_duplicated)``.
    """
    conn.execute("UPDATE photos SET duplicate_of = NULL, content_hash = NULL")

    groups = conn.execute(
        """
        SELECT quick_hash FROM photos
        WHERE quick_hash IS NOT NULL AND kind NOT IN (?, ?, ?)
        GROUP BY quick_hash HAVING COUNT(*) > 1
        """,
        (Kind.VIDEO, Kind.UNREADABLE, Kind.UNSUPPORTED),
    ).fetchall()

    duplicate_count = 0
    duplicate_bytes = 0

    for group in groups:
        rows = conn.execute(
            "SELECT id, path, size_bytes, mtime FROM photos WHERE quick_hash = ? "
            "ORDER BY mtime ASC, path ASC",
            (group["quick_hash"],),
        ).fetchall()

        # A quick hash can collide, so confirm with the full digest before discarding.
        by_content: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            try:
                digest = content_hash(Path(row["path"]))
            except OSError:
                continue
            conn.execute("UPDATE photos SET content_hash = ? WHERE id = ?", (digest, row["id"]))
            by_content.setdefault(digest, []).append(row)

        for identical in by_content.values():
            canonical = identical[0]
            for row in identical[1:]:
                conn.execute(
                    "UPDATE photos SET duplicate_of = ? WHERE id = ?", (canonical["id"], row["id"])
                )
                duplicate_count += 1
                duplicate_bytes += int(row["size_bytes"])

    conn.commit()
    return duplicate_count, duplicate_bytes
