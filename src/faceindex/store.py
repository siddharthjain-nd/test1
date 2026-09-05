"""SQLite store for the photo and face index.

Single-file database, trivially resettable, resumable. Migrates to Postgres in Phase 6.

Schema notes
    A photo's identity is its **content hash**, not its path, so moves and duplicate
    backups collapse correctly.

    ``scan_version`` and ``pipeline_version`` are recorded per row so a code change means
    "reprocess rows below version X" rather than "wipe everything and start again".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 3

# Bumped when the scanner's classification or metadata extraction changes in a way that
# invalidates previously stored rows. Raising it forces a rescan of every file.
SCAN_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL UNIQUE,
    rel_path      TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    mtime         REAL    NOT NULL,

    -- (size, head, tail) digest. Cheap; full sha256 only computed on collision.
    quick_hash    TEXT,
    content_hash  TEXT,

    kind          TEXT    NOT NULL,
    reason        TEXT,

    width         INTEGER,
    height        INTEGER,
    image_format  TEXT,

    taken_at      TEXT,
    -- Where taken_at came from: exif_original | exif_digitized | filename |
    -- exif_modified | folder | mtime. NULL when unknown.
    -- Provenance matters: mtime is unreliable on copied backups and must stay separable.
    taken_at_source TEXT,
    -- EXIF tag 306 is a *modification* time, not a capture time. Bulk edits and copies
    -- rewrite it, so it is stored apart from taken_at and ranked below filename dates.
    exif_modified_at TEXT,
    camera_make   TEXT,
    camera_model  TEXT,
    orientation   INTEGER,
    gps_lat       REAL,
    gps_lon       REAL,

    -- NULL for the canonical copy; otherwise photos.id of the copy we keep.
    duplicate_of  INTEGER REFERENCES photos(id),

    scanned_at    TEXT    NOT NULL,
    scan_version  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_photos_quick_hash   ON photos(quick_hash);
CREATE INDEX IF NOT EXISTS idx_photos_content_hash ON photos(content_hash);
CREATE INDEX IF NOT EXISTS idx_photos_kind         ON photos(kind);
CREATE INDEX IF NOT EXISTS idx_photos_taken_at     ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_duplicate_of ON photos(duplicate_of);

CREATE TABLE IF NOT EXISTS faces (
    id             INTEGER PRIMARY KEY,
    photo_id       INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,

    -- Stable within a photo: index 0 is the largest face. Lets a crop be relocated
    -- without depending on row ids surviving a reindex.
    face_index     INTEGER NOT NULL,

    bbox_x1        REAL NOT NULL,
    bbox_y1        REAL NOT NULL,
    bbox_x2        REAL NOT NULL,
    bbox_y2        REAL NOT NULL,
    landmarks      TEXT NOT NULL,   -- JSON [[x,y] x 5], in decoded-image coordinates
    det_score      REAL NOT NULL,

    -- Auto-derived attributes. Present so errors can be sliced by pose, size, quality.
    interocular_px REAL,
    relative_size  REAL,
    yaw_deg        REAL,
    roll_deg       REAL,
    blur           REAL,
    brightness     REAL,
    dark_fraction  REAL,
    bright_fraction REAL,

    -- Scale from the decoded image back to the original file, so face sizes stay
    -- comparable across photos decoded at different resolutions.
    decode_scale   REAL NOT NULL,

    crop_path      TEXT,
    context_path   TEXT,

    detector       TEXT NOT NULL,
    pool_version   TEXT NOT NULL,
    created_at     TEXT NOT NULL,

    UNIQUE(photo_id, face_index)
);

CREATE INDEX IF NOT EXISTS idx_faces_photo   ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_size    ON faces(interocular_px);
CREATE INDEX IF NOT EXISTS idx_faces_version ON faces(pool_version);

CREATE TABLE IF NOT EXISTS photo_pool_status (
    photo_id     INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    pool_version TEXT NOT NULL,
    n_faces      INTEGER NOT NULL,
    error        TEXT,
    processed_at TEXT NOT NULL
);
"""

# Columns added after the first release, applied to existing databases on open.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("taken_at_source", "TEXT"),
    ("exif_modified_at", "TEXT"),
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
    for column, column_type in _MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {column} {column_type}")


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the index, creating and migrating the schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    if not read_only:
        # WAL survives an abrupt kill mid-scan without corrupting the database.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(_SCHEMA)
        _apply_migrations(conn)
        _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()

    return conn


@contextmanager
def open_index(db_path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    _set_meta(conn, key, value)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def already_scanned(conn: sqlite3.Connection) -> dict[str, tuple[int, float]]:
    """Map ``path -> (size, mtime)`` for rows scanned by the current scanner version.

    Used to resume: a file is reprocessed if it is absent, has changed on disk, or was
    recorded by an older scanner version.
    """
    rows = conn.execute(
        "SELECT path, size_bytes, mtime FROM photos WHERE scan_version = ?",
        (SCAN_VERSION,),
    ).fetchall()
    return {str(r["path"]): (int(r["size_bytes"]), float(r["mtime"])) for r in rows}


def count_photos(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()
    return int(row["n"])
