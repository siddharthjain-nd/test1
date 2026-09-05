"""Date recovery from filenames and folders.

Messaging apps strip EXIF but keep the date in the filename, which is where most of a
typical library's undated photos get their dates back from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceindex import ingest, store
from faceindex.ingest import DateSource


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("IMG-20230115-WA0001.jpg", "2023-01-15T00:00:00"),
        ("IMG_20230115_143022.jpg", "2023-01-15T14:30:22"),
        ("PXL_20220704_101530123.jpg", "2022-07-04T10:15:30"),
        ("20180229_120000.jpg", None),  # 2018 was not a leap year
        ("20200229_120000.jpg", "2020-02-29T12:00:00"),
        ("photo_2021-12-25_18-45-01.jpg", "2021-12-25T18:45:01"),
        ("Screenshot_2024-03-09-07-01-59.png", "2024-03-09T07:01:59"),
        ("IMG-20161131-WA0002.jpg", None),  # November has 30 days
        ("IMG-18991201-WA0001.jpg", None),  # implausibly old
        ("IMG_1234.jpg", None),
        ("DSC_0001.JPG", None),
        ("holiday.jpg", None),
        ("20231301_120000.jpg", None),  # month 13
    ],
)
def test_date_from_filename(filename: str, expected: str | None) -> None:
    assert ingest.date_from_filename(filename) == expected


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("2016 Goa/IMG_0001.jpg", "2016-01-01T00:00:00"),
        ("Trips/2011/beach.jpg", "2011-01-01T00:00:00"),
        ("Photos/Unsorted/x.jpg", None),
        ("1899 Antique/x.jpg", None),
        # The filename itself must never be read as a folder year.
        ("Unsorted/20180101_x.jpg", None),
    ],
)
def test_year_from_folder(rel_path: str, expected: str | None) -> None:
    assert ingest.year_from_folder(rel_path) == expected


def _insert(conn: object, **columns: object) -> None:
    keys = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    conn.execute(  # type: ignore[attr-defined]
        f"INSERT INTO photos ({keys}) VALUES ({marks})", tuple(columns.values())
    )


def test_backfill_priority_order(tmp_path: Path) -> None:
    """EXIF capture > filename > EXIF modified (tag 306) > folder year."""
    db_path = tmp_path / "index.db"
    common = {"size_bytes": 1, "scanned_at": "now", "scan_version": store.SCAN_VERSION}

    with store.open_index(db_path) as conn:
        # Capture-tag EXIF already set during scanning; must never be overwritten.
        _insert(
            conn,
            path="/a.jpg",
            rel_path="2016 Goa/IMG_20230115_120000.jpg",
            mtime=0.0,
            kind="photo",
            taken_at="2019-05-05T00:00:00",
            taken_at_source=DateSource.EXIF_ORIGINAL,
            **common,
        )
        # Filename must beat the modification tag.
        _insert(
            conn,
            path="/b.jpg",
            rel_path="2016 Goa/IMG-20230115-WA0001.jpg",
            mtime=0.0,
            kind="forwarded",
            exif_modified_at="2024-09-09T14:58:49",
            **common,
        )
        # No filename date, so the modification tag is used, but beats the folder year.
        _insert(
            conn,
            path="/c.jpg",
            rel_path="2016 Goa/x.jpg",
            mtime=0.0,
            kind="photo",
            exif_modified_at="2020-02-02T10:00:00",
            **common,
        )
        _insert(conn, path="/d.jpg", rel_path="2016 Goa/y.jpg", mtime=0.0, kind="photo", **common)
        _insert(conn, path="/e.jpg", rel_path="none/z.jpg", mtime=0.0, kind="photo", **common)
        # Non-image kinds must never be dated.
        _insert(
            conn,
            path="/f.mp3",
            rel_path="none/IMG_20230115_120000.mp3",
            mtime=0.0,
            kind="audio",
            **common,
        )
        conn.commit()

        counts = ingest.backfill_dates(conn)

        rows = {
            str(r["path"]): r
            for r in conn.execute("SELECT path, taken_at, taken_at_source FROM photos")
        }

    assert rows["/a.jpg"]["taken_at"] == "2019-05-05T00:00:00"
    assert rows["/a.jpg"]["taken_at_source"] == DateSource.EXIF_ORIGINAL

    assert rows["/b.jpg"]["taken_at"] == "2023-01-15T00:00:00"
    assert rows["/b.jpg"]["taken_at_source"] == DateSource.FILENAME

    assert rows["/c.jpg"]["taken_at"] == "2020-02-02T10:00:00"
    assert rows["/c.jpg"]["taken_at_source"] == DateSource.EXIF_MODIFIED

    assert rows["/d.jpg"]["taken_at"] == "2016-01-01T00:00:00"
    assert rows["/d.jpg"]["taken_at_source"] == DateSource.FOLDER

    assert rows["/e.jpg"]["taken_at"] is None
    assert rows["/f.mp3"]["taken_at"] is None, "audio must never be dated"

    assert counts == {
        DateSource.FILENAME: 1,
        DateSource.EXIF_MODIFIED: 1,
        DateSource.FOLDER: 1,
        DateSource.MTIME: 0,
    }


def test_filename_exif_disagreement_is_detected(tmp_path: Path) -> None:
    """Bulk-rewritten EXIF timestamps show up as filename/EXIF divergence."""
    db_path = tmp_path / "index.db"
    common = {"size_bytes": 1, "scanned_at": "now", "scan_version": store.SCAN_VERSION}

    with store.open_index(db_path) as conn:
        _insert(
            conn,
            path="/agree.jpg",
            rel_path="x/IMG_20250806_234843.jpg",
            mtime=0.0,
            kind="photo",
            taken_at="2025-08-06T23:48:43",
            taken_at_source=DateSource.EXIF_ORIGINAL,
            **common,
        )
        _insert(
            conn,
            path="/disagree.jpg",
            rel_path="x/IMG_20250806_235203.jpg",
            mtime=0.0,
            kind="photo",
            taken_at="2024-01-04T14:58:49",
            taken_at_source=DateSource.EXIF_ORIGINAL,
            **common,
        )
        conn.commit()

        disagreements, comparable = ingest.filename_exif_disagreements(conn)

    assert comparable == 2
    assert disagreements == 1


def test_mtime_fallback_is_opt_in(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    common = {"size_bytes": 1, "scanned_at": "now", "scan_version": store.SCAN_VERSION}

    with store.open_index(db_path) as conn:
        _insert(
            conn,
            path="/e.jpg",
            rel_path="none/e.jpg",
            mtime=1_600_000_000.0,
            kind="photo",
            **common,
        )
        conn.commit()

        assert ingest.backfill_dates(conn)[DateSource.MTIME] == 0
        assert conn.execute("SELECT taken_at FROM photos").fetchone()["taken_at"] is None

        assert ingest.backfill_dates(conn, use_mtime=True)[DateSource.MTIME] == 1
        row = conn.execute("SELECT taken_at, taken_at_source FROM photos").fetchone()

    assert str(row["taken_at"]).startswith("2020-09")
    assert row["taken_at_source"] == DateSource.MTIME


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with store.open_index(db_path) as conn:
        _insert(
            conn,
            path="/f.jpg",
            rel_path="x/IMG_20230115_143022.jpg",
            mtime=0.0,
            kind="photo",
            size_bytes=1,
            scanned_at="now",
            scan_version=store.SCAN_VERSION,
        )
        conn.commit()

        first = ingest.backfill_dates(conn)
        second = ingest.backfill_dates(conn)

    assert first[DateSource.FILENAME] == 1
    assert second[DateSource.FILENAME] == 0, "already-dated rows must not be reprocessed"
