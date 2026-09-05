"""Scanner tests against a synthetic corpus.

The real library lives on the Linux box, so fixtures are generated here: real JPEG/PNG
bytes with real EXIF, plus the awkward cases a 134 GB personal library actually contains
(duplicates, videos, zero-byte and truncated files, spaces in paths).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import ExifTags, Image

from faceindex import ingest, report, store
from faceindex.ingest import Kind


def _write_jpeg(
    path: Path,
    size: tuple[int, int] = (640, 480),
    *,
    colour: tuple[int, int, int] = (120, 90, 60),
    make: str | None = "TestMake",
    model: str | None = "TestModel",
    taken: str | None = "2020:05:17 12:34:56",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)

    exif = Image.Exif()
    if make:
        exif[271] = make
    if model:
        exif[272] = model
    if taken:
        # Main-IFD DateTime; the scanner also reads the Exif sub-IFD variants.
        exif[306] = taken
        exif.get_ifd(ExifTags.IFD.Exif)[36867] = taken

    image.save(path, "JPEG", exif=exif, quality=90)
    return path


def _write_png(path: Path, size: tuple[int, int] = (1080, 1920)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path, "PNG")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A miniature library, including a folder name with a space."""
    root = tmp_path / "photo Timeline"

    # Distinct colours matter: identical pixels give byte-identical JPEGs, which the
    # deduplicator would correctly collapse, skewing the counts asserted below.
    _write_jpeg(root / "2020 Trip" / "IMG_0001.jpg", taken="2020:05:17 12:34:56")
    _write_jpeg(
        root / "2023 Family" / "IMG_0002.jpg",
        colour=(30, 140, 200),
        taken="2023:11:02 08:15:00",
    )
    _write_jpeg(root / "2023 Family" / "IMG_0003.jpg", colour=(5, 5, 5), taken=None)

    # Byte-identical duplicate of IMG_0001 in another folder, as a backup would produce.
    duplicate = root / "Backup 2024" / "IMG_0001.jpg"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes((root / "2020 Trip" / "IMG_0001.jpg").read_bytes())

    _write_jpeg(
        root / "Screenshots" / "Screenshot_20230101.jpg",
        colour=(200, 10, 10),
        make=None,
        model=None,
        taken=None,
    )
    _write_png(root / "Screenshots" / "capture.png")
    _write_jpeg(
        root / "WhatsApp" / "IMG-20230115-WA0001.jpg",
        colour=(10, 200, 10),
        make=None,
        model=None,
        taken=None,
    )
    _write_jpeg(
        root / "thumbs" / "tiny.jpg",
        size=(40, 40),
        colour=(90, 90, 200),
        make=None,
        model=None,
        taken=None,
    )

    (root / "Videos").mkdir(parents=True, exist_ok=True)
    (root / "Videos" / "clip.mp4").write_bytes(b"\x00" * 4096)
    (root / "Videos" / "clip.MOV").write_bytes(b"\x00" * 4096)

    (root / "junk").mkdir(parents=True, exist_ok=True)
    (root / "junk" / "empty.jpg").write_bytes(b"")
    (root / "junk" / "truncated.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (root / "junk" / "notes.txt").write_text("hello")
    (root / "junk" / "photo.dng").write_bytes(b"\x00" * 2048)

    (root / ".hidden").mkdir(parents=True, exist_ok=True)
    (root / ".hidden" / "secret.jpg").write_bytes(b"\x00" * 128)

    return root


def _scan(corpus: Path, tmp_path: Path) -> tuple[dict[str, int], Path]:
    db_path = tmp_path / "index.db"
    with store.open_index(db_path) as conn:
        stats = ingest.scan(conn, corpus)
        ingest.resolve_duplicates(conn)
    return stats.by_kind, db_path


def test_classification_of_every_file_type(corpus: Path, tmp_path: Path) -> None:
    by_kind, db_path = _scan(corpus, tmp_path)

    assert by_kind.get(Kind.VIDEO) == 2, "both video extensions rejected on extension alone"
    assert by_kind.get(Kind.RAW) == 1
    assert by_kind.get(Kind.UNSUPPORTED) == 1, "notes.txt"
    assert by_kind.get(Kind.TINY) == 1
    assert by_kind.get(Kind.FORWARDED) == 1
    assert by_kind.get(Kind.SCREENSHOT) == 2, "filename match plus PNG-without-EXIF"
    assert by_kind.get(Kind.UNREADABLE) == 2, "zero-byte and truncated"
    assert by_kind.get(Kind.PHOTO) == 4, "three originals plus the duplicate copy"

    with store.open_index(db_path, read_only=True) as conn:
        hidden = conn.execute("SELECT COUNT(*) AS n FROM photos WHERE path LIKE '%.hidden%'")
        assert hidden.fetchone()["n"] == 0, "hidden directories must be skipped"


def test_videos_are_never_opened(corpus: Path) -> None:
    """A video must be classified without reading its contents."""
    record = ingest.build_record(corpus / "Videos" / "clip.mp4", corpus)
    assert record.kind == Kind.VIDEO
    assert record.quick_hash is None
    assert record.width is None


def test_exif_is_extracted(corpus: Path, tmp_path: Path) -> None:
    _, db_path = _scan(corpus, tmp_path)
    with store.open_index(db_path, read_only=True) as conn:
        row = conn.execute(
            "SELECT * FROM photos WHERE rel_path LIKE '%IMG_0001.jpg' AND duplicate_of IS NULL"
        ).fetchone()

    assert row["camera_make"] == "TestMake"
    assert row["camera_model"] == "TestModel"
    assert row["taken_at"] == "2020-05-17T12:34:56"
    assert (row["width"], row["height"]) == (640, 480)
    assert row["image_format"] == "JPEG"


def test_duplicates_are_detected_and_one_copy_kept(corpus: Path, tmp_path: Path) -> None:
    _, db_path = _scan(corpus, tmp_path)
    with store.open_index(db_path, read_only=True) as conn:
        rows = conn.execute(
            "SELECT path, duplicate_of, content_hash FROM photos WHERE rel_path LIKE '%IMG_0001%'"
        ).fetchall()

    assert len(rows) == 2
    canonical = [r for r in rows if r["duplicate_of"] is None]
    duplicates = [r for r in rows if r["duplicate_of"] is not None]

    assert len(canonical) == 1, "exactly one copy survives as canonical"
    assert len(duplicates) == 1
    assert canonical[0]["content_hash"] == duplicates[0]["content_hash"]


def test_scan_is_resumable(corpus: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with store.open_index(db_path) as conn:
        first = ingest.scan(conn, corpus)
    with store.open_index(db_path) as conn:
        second = ingest.scan(conn, corpus)

    assert first.processed > 0
    assert second.processed == 0, "an unchanged corpus must be fully skipped"
    assert second.skipped_resumed == first.processed


def test_modified_file_is_rescanned(corpus: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with store.open_index(db_path) as conn:
        ingest.scan(conn, corpus)

    _write_jpeg(corpus / "2020 Trip" / "IMG_0001.jpg", size=(800, 600), taken="2021:01:01 00:00:00")

    with store.open_index(db_path) as conn:
        stats = ingest.scan(conn, corpus)
        row = conn.execute(
            "SELECT width, taken_at FROM photos WHERE rel_path LIKE '2020 Trip%'"
        ).fetchone()

    assert stats.processed == 1
    assert row["width"] == 800
    assert row["taken_at"] == "2021-01-01T00:00:00"


def test_quick_hash_distinguishes_content(tmp_path: Path) -> None:
    a = _write_jpeg(tmp_path / "a.jpg", colour=(10, 20, 30))
    b = _write_jpeg(tmp_path / "b.jpg", colour=(200, 100, 50))
    c = tmp_path / "c.jpg"
    c.write_bytes(a.read_bytes())

    assert ingest.quick_hash(a, a.stat().st_size) == ingest.quick_hash(c, c.stat().st_size)
    assert ingest.quick_hash(a, a.stat().st_size) != ingest.quick_hash(b, b.stat().st_size)
    assert ingest.content_hash(a) == ingest.content_hash(c)


def test_face_candidates_exclude_duplicates_and_junk(corpus: Path, tmp_path: Path) -> None:
    _, db_path = _scan(corpus, tmp_path)
    placeholders = ",".join("?" for _ in ingest.FACE_CANDIDATE_KINDS)
    with store.open_index(db_path, read_only=True) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM photos WHERE kind IN ({placeholders}) "
            f"AND duplicate_of IS NULL",
            tuple(sorted(ingest.FACE_CANDIDATE_KINDS)),
        ).fetchone()

    assert row["n"] == 4, "3 unique photos + 1 forwarded; duplicate and junk excluded"


def test_report_runs(corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, db_path = _scan(corpus, tmp_path)
    with store.open_index(db_path, read_only=True) as conn:
        report.print_report(conn)

    output = capsys.readouterr().out
    assert "Corpus composition" in output
    assert "Unique face candidates" in output


def test_missing_file_does_not_crash_the_scan(tmp_path: Path) -> None:
    record = ingest.build_record(tmp_path / "does_not_exist.jpg", tmp_path)
    assert record.kind == Kind.UNREADABLE
    assert record.reason is not None and "stat failed" in record.reason
