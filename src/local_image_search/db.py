from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from local_image_search.models import ImageFile, IndexedImage

SCHEMA_VERSION = 1


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            created_at REAL,
            modified_at REAL NOT NULL,
            caption TEXT NOT NULL DEFAULT '',
            caption_model TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            embedding_json TEXT NOT NULL DEFAULT '',
            thumbnail_path TEXT,
            indexed_at REAL NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_images_modified_at ON images(modified_at);
        """
    )
    _add_column_if_missing(conn, "images", "thumbnail_path", "TEXT")
    conn.execute(
        """
        INSERT INTO schema_meta (key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def needs_indexing(
    conn: sqlite3.Connection,
    image: ImageFile,
    caption_model: str,
    embedding_model: str,
) -> bool:
    row = conn.execute(
        """
        SELECT file_size, modified_at, caption_model, embedding_model, caption, embedding_json
        FROM images
        WHERE path = ?
        """,
        (str(image.path),),
    ).fetchone()
    if row is None:
        return True
    return (
        row["file_size"] != image.file_size
        or row["modified_at"] != image.modified_at
        or row["caption_model"] != caption_model
        or row["embedding_model"] != embedding_model
        or not row["caption"]
        or not row["embedding_json"]
    )


def upsert_indexed_image(
    conn: sqlite3.Connection,
    image: ImageFile,
    caption: str,
    caption_model: str,
    embedding_model: str,
    embedding: list[float],
    thumbnail_path: Path | None,
) -> None:
    conn.execute(
        """
        INSERT INTO images (
            path,
            file_name,
            file_size,
            created_at,
            modified_at,
            caption,
            caption_model,
            embedding_model,
            embedding_json,
            thumbnail_path,
            indexed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            file_name = excluded.file_name,
            file_size = excluded.file_size,
            created_at = excluded.created_at,
            modified_at = excluded.modified_at,
            caption = excluded.caption,
            caption_model = excluded.caption_model,
            embedding_model = excluded.embedding_model,
            embedding_json = excluded.embedding_json,
            thumbnail_path = excluded.thumbnail_path,
            indexed_at = excluded.indexed_at
        """,
        (
            str(image.path),
            image.file_name,
            image.file_size,
            image.created_at,
            image.modified_at,
            caption,
            caption_model,
            embedding_model,
            json.dumps(embedding),
            str(thumbnail_path.resolve()) if thumbnail_path else None,
            time.time(),
        ),
    )


def list_indexed_images(conn: sqlite3.Connection) -> list[IndexedImage]:
    rows = conn.execute(
        """
        SELECT id, path, file_name, file_size, created_at, modified_at,
               caption, caption_model, embedding_model, embedding_json, thumbnail_path
        FROM images
        WHERE caption != '' AND embedding_json != ''
        ORDER BY path
        """
    ).fetchall()
    return [_row_to_indexed_image(row) for row in rows]


def delete_missing_paths(conn: sqlite3.Connection, seen_paths: Iterable[Path]) -> int:
    seen = {str(path) for path in seen_paths}
    rows = conn.execute("SELECT id, path FROM images").fetchall()
    deleted = 0
    for row in rows:
        if row["path"] not in seen:
            conn.execute("DELETE FROM images WHERE id = ?", (row["id"],))
            deleted += 1
    return deleted


def count_images(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()
    return int(row["count"])


def get_thumbnail_path(conn: sqlite3.Connection, image_path: Path) -> Path | None:
    row = conn.execute(
        "SELECT thumbnail_path FROM images WHERE path = ?",
        (str(image_path),),
    ).fetchone()
    if row is None or row["thumbnail_path"] is None:
        return None
    return Path(row["thumbnail_path"])


def update_thumbnail_path(
    conn: sqlite3.Connection,
    image_path: Path,
    thumbnail_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE images
        SET thumbnail_path = ?
        WHERE path = ?
        """,
        (str(thumbnail_path.resolve()), str(image_path)),
    )


def _row_to_indexed_image(row: sqlite3.Row) -> IndexedImage:
    return IndexedImage(
        id=int(row["id"]),
        path=Path(row["path"]),
        file_name=str(row["file_name"]),
        file_size=int(row["file_size"]),
        created_at=row["created_at"],
        modified_at=float(row["modified_at"]),
        caption=str(row["caption"]),
        caption_model=str(row["caption_model"]),
        embedding_model=str(row["embedding_model"]),
        embedding=[float(value) for value in json.loads(row["embedding_json"])],
        thumbnail_path=_normalize_thumbnail_path(row["thumbnail_path"]),
    )


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _normalize_thumbnail_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()
