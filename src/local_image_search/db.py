from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from local_image_search.models import ImageFile, IndexedImage, SearchResult

SQLITE_TIMEOUT_SECONDS = 30
VECTOR_TABLE_NAME = "image_embeddings"
VECTOR_DIMENSIONS = 512


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={SQLITE_TIMEOUT_SECONDS * 1000}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    load_sqlite_vec(conn)
    return conn


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={SQLITE_TIMEOUT_SECONDS * 1000}")
    load_sqlite_vec(conn)
    return conn


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            created_at REAL,
            modified_at REAL NOT NULL,
            embedding_model TEXT NOT NULL DEFAULT '',
            thumbnail_path TEXT,
            indexed_at REAL NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_images_modified_at ON images(modified_at);
        """
    )
    conn.commit()


def ensure_vector_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE_NAME}
        USING vec0(embedding float[{VECTOR_DIMENSIONS}])
        """
    )


def needs_indexing(
    conn: sqlite3.Connection,
    image: ImageFile,
    embedding_model: str,
) -> bool:
    vector_exists_sql = _vector_exists_sql(conn)
    row = conn.execute(
        f"""
        SELECT file_size, modified_at, embedding_model,
               {vector_exists_sql} AS has_embedding
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
        or row["embedding_model"] != embedding_model
        or not row["has_embedding"]
    )


def upsert_indexed_image(
    conn: sqlite3.Connection,
    image: ImageFile,
    embedding_model: str,
    embedding: list[float],
    thumbnail_path: Path | None,
) -> None:
    _validate_embedding_dimensions(embedding)
    conn.execute(
        """
        INSERT INTO images (
            path,
            file_name,
            file_size,
            created_at,
            modified_at,
            embedding_model,
            thumbnail_path,
            indexed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            file_name = excluded.file_name,
            file_size = excluded.file_size,
            created_at = excluded.created_at,
            modified_at = excluded.modified_at,
            embedding_model = excluded.embedding_model,
            thumbnail_path = excluded.thumbnail_path,
            indexed_at = excluded.indexed_at
        """,
        (
            str(image.path),
            image.file_name,
            image.file_size,
            image.created_at,
            image.modified_at,
            embedding_model,
            str(thumbnail_path.resolve()) if thumbnail_path else None,
            time.time(),
        ),
    )
    image_id = get_image_id(conn, image.path)
    conn.execute(
        f"DELETE FROM {VECTOR_TABLE_NAME} WHERE rowid = ?",
        (image_id,),
    )
    conn.execute(
        f"INSERT INTO {VECTOR_TABLE_NAME} (rowid, embedding) VALUES (?, ?)",
        (image_id, serialize_embedding(embedding)),
    )


def get_image_id(conn: sqlite3.Connection, image_path: Path) -> int:
    row = conn.execute(
        "SELECT id FROM images WHERE path = ?",
        (str(image_path),),
    ).fetchone()
    if row is None:
        raise ValueError(f"Image was not found after upsert: {image_path}")
    return int(row["id"])


def search_indexed_images(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    embedding_model: str,
    limit: int,
    exclude_path: Path | None = None,
) -> list[SearchResult]:
    _validate_embedding_dimensions(query_embedding)
    if not vector_table_exists(conn):
        return []
    search_limit = limit + 1 if exclude_path else limit
    rows = conn.execute(
        f"""
        SELECT images.id, images.path, images.file_name, images.file_size,
               images.created_at, images.modified_at, images.embedding_model,
               images.thumbnail_path,
               matches.distance,
               (1.0 - ((matches.distance * matches.distance) / 2.0)) AS score
        FROM {VECTOR_TABLE_NAME} AS matches
        JOIN images ON images.id = matches.rowid
        WHERE matches.embedding MATCH ?
          AND matches.k = ?
          AND images.embedding_model = ?
        ORDER BY matches.distance
        """,
        (serialize_embedding(query_embedding), search_limit, embedding_model),
    ).fetchall()
    excluded = _normalize_search_path(exclude_path)
    results = []
    for row in rows:
        image = _row_to_indexed_image(row)
        if excluded is not None and _normalize_search_path(image.path) == excluded:
            continue
        results.append(
            SearchResult(
                image=image,
                score=float(row["score"]),
            )
        )
        if len(results) >= limit:
            break
    return results


def delete_missing_paths(conn: sqlite3.Connection, seen_paths: Iterable[Path]) -> int:
    seen = {str(path) for path in seen_paths}
    rows = conn.execute("SELECT id, path FROM images").fetchall()
    deleted = 0
    for row in rows:
        if row["path"] not in seen:
            if vector_table_exists(conn):
                conn.execute(
                    f"DELETE FROM {VECTOR_TABLE_NAME} WHERE rowid = ?",
                    (row["id"],),
                )
            conn.execute("DELETE FROM images WHERE id = ?", (row["id"],))
            deleted += 1
    return deleted


def count_images(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()
    return int(row["count"])


def count_searchable_images(
    conn: sqlite3.Connection,
    embedding_model: str,
) -> int:
    if not vector_table_exists(conn):
        return 0
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM {VECTOR_TABLE_NAME}
        JOIN images ON images.id = {VECTOR_TABLE_NAME}.rowid
        WHERE images.embedding_model = ?
        """,
        (embedding_model,),
    ).fetchone()
    return int(row["count"])


def index_version(conn: sqlite3.Connection) -> tuple[int, float]:
    if not vector_table_exists(conn):
        return 0, 0
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count, COALESCE(MAX(indexed_at), 0) AS latest_indexed_at
        FROM {VECTOR_TABLE_NAME}
        JOIN images ON images.id = {VECTOR_TABLE_NAME}.rowid
        """
    ).fetchone()
    return int(row["count"]), float(row["latest_indexed_at"])


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


def serialize_embedding(embedding: list[float]) -> bytes:
    import sqlite_vec

    return sqlite_vec.serialize_float32(embedding)


def vector_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (VECTOR_TABLE_NAME,),
    ).fetchone()
    return row is not None


def _row_to_indexed_image(row: sqlite3.Row) -> IndexedImage:
    return IndexedImage(
        id=int(row["id"]),
        path=Path(row["path"]),
        file_name=str(row["file_name"]),
        file_size=int(row["file_size"]),
        created_at=row["created_at"],
        modified_at=float(row["modified_at"]),
        embedding_model=str(row["embedding_model"]),
        thumbnail_path=_normalize_thumbnail_path(row["thumbnail_path"]),
    )


def _vector_exists_sql(conn: sqlite3.Connection) -> str:
    if not vector_table_exists(conn):
        return "0"
    return (
        "EXISTS ("
        f"SELECT 1 FROM {VECTOR_TABLE_NAME} "
        f"WHERE {VECTOR_TABLE_NAME}.rowid = images.id"
        ")"
    )


def _normalize_thumbnail_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _normalize_search_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.expanduser().resolve()


def _validate_embedding_dimensions(embedding: list[float]) -> None:
    if len(embedding) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Expected {VECTOR_DIMENSIONS} embedding dimensions, got {len(embedding)}"
        )
