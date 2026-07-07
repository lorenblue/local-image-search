from __future__ import annotations

import sqlite3
from pathlib import Path

from local_image_search.db import (
    FACE_VECTOR_TABLE_NAME,
    connect,
    connect_readonly,
    count_face_embeddings,
    count_faces,
    count_images,
    count_searchable_images,
    delete_missing_paths,
    ensure_face_vector_table,
    ensure_vector_table,
    get_image_id,
    get_primary_face_for_image,
    init_db,
    list_faces_for_image,
    search_similar_faces,
    serialize_embedding,
    upsert_faces_for_image,
    upsert_indexed_image,
)
from local_image_search.models import FaceBox, ImageFile


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
        "face_detection_model",
        "faces_indexed_at",
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
        "detection_model",
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


def test_upsert_faces_for_image_replaces_existing_face_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    image_path = tmp_path / "portrait.jpg"
    image = ImageFile(image_path, image_path.name, 10, None, 1)

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        image_id = upsert_indexed_image(
            conn,
            image,
            "test-clip",
            [1.0] + [0.0] * 511,
            None,
        )

        upsert_faces_for_image(
            conn,
            image_id,
            "detector-a",
            [
                FaceBox(
                    x=1,
                    y=2,
                    width=3,
                    height=4,
                    detection_score=0.9,
                    embedding=[1.0] + [0.0] * 511,
                ),
                FaceBox(x=5, y=6, width=7, height=8, detection_score=None),
            ],
        )
        upsert_faces_for_image(
            conn,
            image_id,
            "detector-b",
            [FaceBox(x=9, y=10, width=11, height=12, detection_score=0.8)],
        )
        conn.commit()

        faces = list_faces_for_image(conn, get_image_id(conn, image_path))
        image_row = conn.execute(
            "SELECT face_detection_model, faces_indexed_at FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        face_row = conn.execute(
            "SELECT embedding_model FROM faces WHERE image_id = ?",
            (image_id,),
        ).fetchone()

        assert count_faces(conn) == 1
        assert count_face_embeddings(conn) == 0
        assert faces == [FaceBox(x=9, y=10, width=11, height=12, detection_score=0.8)]
        assert face_row["embedding_model"] == ""
        assert image_row["face_detection_model"] == "detector-b"
        assert image_row["faces_indexed_at"] > 0


def test_upsert_faces_for_image_stores_face_embedding_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    image_path = tmp_path / "portrait.jpg"
    image = ImageFile(image_path, image_path.name, 10, None, 1)

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        image_id = upsert_indexed_image(
            conn,
            image,
            "test-clip",
            [1.0] + [0.0] * 511,
            None,
        )

        upsert_faces_for_image(
            conn,
            image_id,
            "test-face",
            [
                FaceBox(
                    x=1,
                    y=2,
                    width=3,
                    height=4,
                    detection_score=0.9,
                    embedding=[0.0, 1.0] + [0.0] * 510,
                )
            ],
        )
        conn.commit()

        face_row = conn.execute(
            "SELECT id, embedding_model FROM faces WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        vector_row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {FACE_VECTOR_TABLE_NAME} WHERE rowid = ?",
            (face_row["id"],),
        ).fetchone()

        assert face_row["embedding_model"] == "test-face"
        assert count_face_embeddings(conn) == 1
        assert vector_row["count"] == 1


def test_search_similar_faces_returns_nearest_face_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    source_path = tmp_path / "source.jpg"
    near_path = tmp_path / "near.jpg"
    far_path = tmp_path / "far.jpg"
    for path in [source_path, near_path, far_path]:
        path.write_bytes(b"test image placeholder")

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        source_image_id = _insert_indexed_image(conn, source_path)
        near_image_id = _insert_indexed_image(conn, near_path)
        far_image_id = _insert_indexed_image(conn, far_path)
        upsert_faces_for_image(
            conn,
            source_image_id,
            "test-face",
            [FaceBox(1, 2, 3, 4, 0.99, embedding=[1.0] + [0.0] * 511)],
        )
        upsert_faces_for_image(
            conn,
            near_image_id,
            "test-face",
            [FaceBox(5, 6, 7, 8, 0.98, embedding=_normalize([0.9, 0.1] + [0.0] * 510))],
        )
        upsert_faces_for_image(
            conn,
            far_image_id,
            "test-face",
            [FaceBox(9, 10, 11, 12, 0.97, embedding=[0.0, 1.0] + [0.0] * 510)],
        )
        conn.commit()

        source_face_id = conn.execute(
            "SELECT id FROM faces WHERE image_id = ?",
            (source_image_id,),
        ).fetchone()["id"]
        near_face_id = conn.execute(
            "SELECT id FROM faces WHERE image_id = ?",
            (near_image_id,),
        ).fetchone()["id"]
        far_face_id = conn.execute(
            "SELECT id FROM faces WHERE image_id = ?",
            (far_image_id,),
        ).fetchone()["id"]

        results = search_similar_faces(conn, source_face_id, limit=2)

        assert [result.face.id for result in results] == [near_face_id, far_face_id]
        assert [result.face.image.path for result in results] == [near_path, far_path]
        assert results[0].score > results[1].score


def test_search_similar_faces_returns_one_best_match_per_image(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    source_path = tmp_path / "source.jpg"
    near_path = tmp_path / "near.jpg"
    far_path = tmp_path / "far.jpg"
    for path in [source_path, near_path, far_path]:
        path.write_bytes(b"test image placeholder")

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        source_image_id = _insert_indexed_image(conn, source_path)
        near_image_id = _insert_indexed_image(conn, near_path)
        far_image_id = _insert_indexed_image(conn, far_path)
        upsert_faces_for_image(
            conn,
            source_image_id,
            "test-face",
            [
                FaceBox(1, 2, 3, 4, 0.99, embedding=[1.0] + [0.0] * 511),
                FaceBox(2, 3, 4, 5, 0.98, embedding=_normalize([0.99, 0.01] + [0.0] * 510)),
            ],
        )
        upsert_faces_for_image(
            conn,
            near_image_id,
            "test-face",
            [
                FaceBox(5, 6, 7, 8, 0.97, embedding=_normalize([0.90, 0.10] + [0.0] * 510)),
                FaceBox(6, 7, 8, 9, 0.96, embedding=_normalize([0.80, 0.20] + [0.0] * 510)),
            ],
        )
        upsert_faces_for_image(
            conn,
            far_image_id,
            "test-face",
            [FaceBox(9, 10, 11, 12, 0.95, embedding=[0.0, 1.0] + [0.0] * 510)],
        )
        conn.commit()

        source_face_id = conn.execute(
            "SELECT id FROM faces WHERE image_id = ? ORDER BY id LIMIT 1",
            (source_image_id,),
        ).fetchone()["id"]

        results = search_similar_faces(conn, source_face_id, limit=3)

        assert [result.face.image.path for result in results] == [near_path, far_path]
        assert len({result.face.image.id for result in results}) == len(results)


def test_get_primary_face_for_image_uses_largest_indexed_face(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    image_path = tmp_path / "group.jpg"
    image_path.write_bytes(b"test image placeholder")

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        image_id = _insert_indexed_image(conn, image_path)
        upsert_faces_for_image(
            conn,
            image_id,
            "test-face",
            [
                FaceBox(1, 2, 10, 10, 0.99, embedding=[1.0] + [0.0] * 511),
                FaceBox(5, 6, 30, 30, 0.80, embedding=[0.0, 1.0] + [0.0] * 510),
            ],
        )
        conn.commit()

        primary = get_primary_face_for_image(conn, image_id)

        assert primary is not None
        assert primary.image.path == image_path
        assert primary.box.x == 5
        assert primary.box.width == 30


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


def _insert_indexed_image(conn: sqlite3.Connection, image_path: Path) -> int:
    return upsert_indexed_image(
        conn,
        ImageFile(
            image_path,
            image_path.name,
            image_path.stat().st_size,
            None,
            image_path.stat().st_mtime,
        ),
        "test-clip",
        [1.0] + [0.0] * 511,
        None,
    )


def _normalize(values: list[float]) -> list[float]:
    magnitude = sum(value * value for value in values) ** 0.5
    return [value / magnitude for value in values]
