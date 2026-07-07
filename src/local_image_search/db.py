from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from local_image_search.models import (
    FaceBox,
    FaceSearchResult,
    ImageFile,
    IndexedFace,
    IndexedImage,
    SearchResult,
)

SQLITE_TIMEOUT_SECONDS = 30
VECTOR_TABLE_NAME = "image_embeddings"
VECTOR_DIMENSIONS = 512
FACE_VECTOR_TABLE_NAME = "face_embeddings"
FACE_VECTOR_DIMENSIONS = 512
SEARCH_OVERFETCH_MULTIPLIER = 5


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
            face_detection_model TEXT NOT NULL DEFAULT '',
            faces_indexed_at REAL NOT NULL DEFAULT 0,
            thumbnail_path TEXT,
            indexed_at REAL NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_images_modified_at ON images(modified_at);

        CREATE TABLE IF NOT EXISTS face_clusters (
            id INTEGER PRIMARY KEY,
            label TEXT,
            created_at REAL NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            detection_score REAL,
            detection_model TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            cluster_id INTEGER REFERENCES face_clusters(id) ON DELETE SET NULL,
            indexed_at REAL NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
        CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
        """
    )
    _ensure_column(
        conn,
        "images",
        "face_detection_model",
        "TEXT NOT NULL DEFAULT ''",
    )
    _ensure_column(
        conn,
        "images",
        "faces_indexed_at",
        "REAL NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "faces",
        "detection_model",
        "TEXT NOT NULL DEFAULT ''",
    )
    conn.commit()


def ensure_vector_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE_NAME}
        USING vec0(embedding float[{VECTOR_DIMENSIONS}])
        """
    )


def ensure_face_vector_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {FACE_VECTOR_TABLE_NAME}
        USING vec0(embedding float[{FACE_VECTOR_DIMENSIONS}])
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


def needs_face_indexing(
    conn: sqlite3.Connection,
    image_path: Path,
    detection_model: str,
    embedding_model: str,
) -> bool:
    row = conn.execute(
        """
        SELECT id, face_detection_model, faces_indexed_at
        FROM images
        WHERE path = ?
        """,
        (str(image_path),),
    ).fetchone()
    if row is None:
        return True
    return (
        row["face_detection_model"] != detection_model
        or float(row["faces_indexed_at"]) <= 0
        or _has_stale_face_embeddings(conn, int(row["id"]), embedding_model)
    )


def _has_stale_face_embeddings(
    conn: sqlite3.Connection,
    image_id: int,
    embedding_model: str,
) -> bool:
    if not face_vector_table_exists(conn):
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM faces WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        return int(row["count"]) > 0

    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM faces
        WHERE faces.image_id = ?
          AND (
            faces.embedding_model != ?
            OR NOT EXISTS (
              SELECT 1
              FROM {FACE_VECTOR_TABLE_NAME}
              WHERE {FACE_VECTOR_TABLE_NAME}.rowid = faces.id
            )
          )
        """,
        (image_id, embedding_model),
    ).fetchone()
    return int(row["count"]) > 0


def upsert_indexed_image(
    conn: sqlite3.Connection,
    image: ImageFile,
    embedding_model: str,
    embedding: list[float],
    thumbnail_path: Path | None,
) -> int:
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
    reset_faces_for_image(conn, image_id)
    return image_id


def upsert_faces_for_image(
    conn: sqlite3.Connection,
    image_id: int,
    detection_model: str,
    faces: list[FaceBox],
    embedding_model: str | None = None,
) -> None:
    reset_faces_for_image(conn, image_id)
    indexed_at = time.time()
    embedding_model = embedding_model or detection_model
    if any(face.embedding is not None for face in faces):
        ensure_face_vector_table(conn)

    for face in faces:
        face_embedding_model = embedding_model if face.embedding is not None else ""
        cursor = conn.execute(
            """
            INSERT INTO faces (
                image_id,
                x,
                y,
                width,
                height,
                detection_score,
                detection_model,
                embedding_model,
                indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                face.x,
                face.y,
                face.width,
                face.height,
                face.detection_score,
                detection_model,
                face_embedding_model,
                indexed_at,
            ),
        )
        if face.embedding is not None:
            _validate_embedding_dimensions(face.embedding)
            conn.execute(
                f"INSERT INTO {FACE_VECTOR_TABLE_NAME} (rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, serialize_embedding(face.embedding)),
            )

    conn.execute(
        """
        UPDATE images
        SET face_detection_model = ?,
            faces_indexed_at = ?
        WHERE id = ?
        """,
        (detection_model, indexed_at, image_id),
    )


def reset_faces_for_image(conn: sqlite3.Connection, image_id: int) -> None:
    delete_faces_for_image(conn, image_id)
    conn.execute(
        """
        UPDATE images
        SET face_detection_model = '',
            faces_indexed_at = 0
        WHERE id = ?
        """,
        (image_id,),
    )


def list_faces_for_image(conn: sqlite3.Connection, image_id: int) -> list[FaceBox]:
    rows = conn.execute(
        """
        SELECT x, y, width, height, detection_score
        FROM faces
        WHERE image_id = ?
        ORDER BY id
        """,
        (image_id,),
    ).fetchall()
    return [
        FaceBox(
            x=float(row["x"]),
            y=float(row["y"]),
            width=float(row["width"]),
            height=float(row["height"]),
            detection_score=(
                None if row["detection_score"] is None else float(row["detection_score"])
            ),
        )
        for row in rows
    ]


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
    if limit <= 0:
        return []
    if not vector_table_exists(conn):
        return []
    search_limit = limit * SEARCH_OVERFETCH_MULTIPLIER
    if exclude_path:
        search_limit += 1
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
        if not image.path.exists():
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


def search_similar_faces(
    conn: sqlite3.Connection,
    face_id: int,
    limit: int,
) -> list[FaceSearchResult]:
    if limit <= 0:
        return []
    if not face_vector_table_exists(conn):
        return []

    source = conn.execute(
        f"""
        SELECT faces.embedding_model, {FACE_VECTOR_TABLE_NAME}.embedding
        FROM faces
        JOIN {FACE_VECTOR_TABLE_NAME} ON {FACE_VECTOR_TABLE_NAME}.rowid = faces.id
        WHERE faces.id = ?
        """,
        (face_id,),
    ).fetchone()
    if source is None:
        raise ValueError(f"Face embedding was not found: {face_id}")

    search_limit = (limit * SEARCH_OVERFETCH_MULTIPLIER) + 1
    rows = conn.execute(
        f"""
        SELECT faces.id AS face_id,
               faces.x,
               faces.y,
               faces.width,
               faces.height,
               faces.detection_score,
               faces.embedding_model AS face_embedding_model,
               images.id,
               images.path,
               images.file_name,
               images.file_size,
               images.created_at,
               images.modified_at,
               images.embedding_model,
               images.thumbnail_path,
               matches.distance,
               (1.0 - ((matches.distance * matches.distance) / 2.0)) AS score
        FROM {FACE_VECTOR_TABLE_NAME} AS matches
        JOIN faces ON faces.id = matches.rowid
        JOIN images ON images.id = faces.image_id
        WHERE matches.embedding MATCH ?
          AND matches.k = ?
          AND faces.embedding_model = ?
        ORDER BY matches.distance
        """,
        (source["embedding"], search_limit, source["embedding_model"]),
    ).fetchall()

    results = []
    for row in rows:
        if int(row["face_id"]) == face_id:
            continue
        image = _row_to_indexed_image(row)
        if not image.path.exists():
            continue
        results.append(
            FaceSearchResult(
                face=_row_to_indexed_face(row, image),
                score=float(row["score"]),
            )
        )
        if len(results) >= limit:
            break
    return results


def delete_missing_paths(
    conn: sqlite3.Connection,
    seen_paths: Iterable[Path],
    roots: Iterable[Path],
) -> int:
    seen = {_normalize_stored_path(path) for path in seen_paths}
    scopes = [_scan_scope(root) for root in roots]
    if not scopes:
        return 0

    rows = conn.execute("SELECT id, path FROM images").fetchall()
    deleted = 0
    for row in rows:
        image_path = _normalize_stored_path(Path(row["path"]))
        if image_path in seen or not _path_is_in_scan_scope(image_path, scopes):
            continue
        if vector_table_exists(conn):
            conn.execute(
                f"DELETE FROM {VECTOR_TABLE_NAME} WHERE rowid = ?",
                (row["id"],),
            )
        delete_faces_for_image(conn, int(row["id"]))
        conn.execute("DELETE FROM images WHERE id = ?", (row["id"],))
        deleted += 1
    return deleted


def delete_faces_for_image(conn: sqlite3.Connection, image_id: int) -> None:
    face_ids = [
        int(row["id"])
        for row in conn.execute(
            "SELECT id FROM faces WHERE image_id = ?",
            (image_id,),
        ).fetchall()
    ]
    if face_ids and face_vector_table_exists(conn):
        conn.executemany(
            f"DELETE FROM {FACE_VECTOR_TABLE_NAME} WHERE rowid = ?",
            [(face_id,) for face_id in face_ids],
        )
    conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))


def count_images(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()
    return int(row["count"])


def count_faces(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM faces").fetchone()
    return int(row["count"])


def count_face_embeddings(conn: sqlite3.Connection) -> int:
    if not face_vector_table_exists(conn):
        return 0
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {FACE_VECTOR_TABLE_NAME}"
    ).fetchone()
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
    return _table_exists(conn, VECTOR_TABLE_NAME)


def face_vector_table_exists(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn, FACE_VECTOR_TABLE_NAME)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


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


def _row_to_indexed_face(row: sqlite3.Row, image: IndexedImage) -> IndexedFace:
    return IndexedFace(
        id=int(row["face_id"]),
        image=image,
        box=FaceBox(
            x=float(row["x"]),
            y=float(row["y"]),
            width=float(row["width"]),
            height=float(row["height"]),
            detection_score=(
                None if row["detection_score"] is None else float(row["detection_score"])
            ),
            id=int(row["face_id"]),
        ),
        embedding_model=str(row["face_embedding_model"]),
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


def _normalize_stored_path(value: Path) -> Path:
    return value.expanduser().resolve(strict=False)


def _scan_scope(root: Path) -> tuple[Path, bool]:
    normalized = root.expanduser().resolve(strict=False)
    return normalized, root.is_file()


def _path_is_in_scan_scope(path: Path, scopes: list[tuple[Path, bool]]) -> bool:
    for root, exact_match in scopes:
        if exact_match:
            if path == root:
                return True
            continue
        if path == root or path.is_relative_to(root):
            return True
    return False


def _validate_embedding_dimensions(embedding: list[float]) -> None:
    if len(embedding) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Expected {VECTOR_DIMENSIONS} embedding dimensions, got {len(embedding)}"
        )
