from __future__ import annotations

import sqlite3
from pathlib import Path

from local_image_search.db import (
    FACE_VECTOR_TABLE_NAME,
    connect,
    connect_readonly,
    count_images,
    count_searchable_images,
    delete_missing_paths,
    ensure_face_vector_table,
    ensure_vector_table,
    init_db,
    serialize_embedding,
    upsert_indexed_image,
)
from local_image_search.models import ImageFile


def test_init_db_uses_image_and_face_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    with connect(db_path) as conn:
        init_db(conn)

        image_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(images)").fetchall()
        }
        face_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(faces)").fetchall()
        }
        face_cluster_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(face_clusters)").fetchall()
        }

    assert image_columns == {
        "id",
        "path",
        "file_name",
        "file_size",
        "created_at",
        "modified_at",
        "embedding_model",
        "thumbnail_path",
        "indexed_at",
    }
    assert face_columns == {
        "id",
        "image_id",
        "x",
        "y",
        "width",
        "height",
        "detection_score",
        "embedding_model",
        "cluster_id",
        "indexed_at",
    }
    assert face_cluster_columns == {
        "id",
        "label",
        "created_at",
        "updated_at",
    }


def test_connect_readonly_reads_without_allowing_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    with connect(db_path) as conn:
        init_db(conn)

    with connect_readonly(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()
        assert row["count"] == 0

        try:
            conn.execute(
                """
                INSERT INTO images (path, file_name, file_size, modified_at)
                VALUES ('/tmp/example.jpg', 'example.jpg', 1, 1)
                """
            )
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower() or "read-only" in str(exc).lower()
        else:
            raise AssertionError("read-only connection unexpectedly allowed a write")


def test_delete_missing_paths_only_prunes_scanned_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    album_a = tmp_path / "album-a"
    album_b = tmp_path / "album-b"
    album_a.mkdir()
    album_b.mkdir()

    live_a = album_a / "live.jpg"
    missing_a = album_a / "deleted.jpg"
    missing_b = album_b / "deleted.jpg"
    live_a.write_bytes(b"test image placeholder")

    embedding_model = "test-clip"
    embedding = [1.0] + [0.0] * 511
    images = [
        ImageFile(live_a, live_a.name, 10, None, 1),
        ImageFile(missing_a, missing_a.name, 10, None, 1),
        ImageFile(missing_b, missing_b.name, 10, None, 1),
    ]

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        for image in images:
            upsert_indexed_image(conn, image, embedding_model, embedding, None)
        deleted = delete_missing_paths(conn, [live_a], [album_a])
        conn.commit()

        rows = conn.execute("SELECT path FROM images ORDER BY path").fetchall()
        paths = {Path(row["path"]) for row in rows}

        assert deleted == 1
        assert paths == {live_a, missing_b}
        assert count_images(conn) == 2
        assert count_searchable_images(conn, embedding_model) == 2


def test_delete_missing_paths_removes_face_metadata_and_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    album = tmp_path / "album"
    album.mkdir()

    missing_path = album / "deleted.jpg"
    embedding_model = "test-clip"
    face_model = "test-face"
    image = ImageFile(missing_path, missing_path.name, 10, None, 1)

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        ensure_face_vector_table(conn)
        upsert_indexed_image(
            conn,
            image,
            embedding_model,
            [1.0] + [0.0] * 511,
            None,
        )
        image_id = conn.execute(
            "SELECT id FROM images WHERE path = ?",
            (str(missing_path),),
        ).fetchone()["id"]
        conn.execute(
            """
            INSERT INTO faces (
                image_id, x, y, width, height, detection_score, embedding_model, indexed_at
            )
            VALUES (?, 1, 2, 3, 4, 0.9, ?, 1)
            """,
            (image_id, face_model),
        )
        face_id = conn.execute("SELECT id FROM faces WHERE image_id = ?", (image_id,)).fetchone()[
            "id"
        ]
        conn.execute(
            f"INSERT INTO {FACE_VECTOR_TABLE_NAME} (rowid, embedding) VALUES (?, ?)",
            (face_id, serialize_embedding([1.0] + [0.0] * 511)),
        )

        deleted = delete_missing_paths(conn, [], [album])
        conn.commit()

        assert deleted == 1
        assert conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()["count"] == 0
        assert conn.execute("SELECT COUNT(*) AS count FROM faces").fetchone()["count"] == 0
        assert conn.execute(f"SELECT COUNT(*) AS count FROM {FACE_VECTOR_TABLE_NAME}").fetchone()[
            "count"
        ] == 0
