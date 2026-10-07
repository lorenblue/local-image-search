from __future__ import annotations

from pathlib import Path

from PIL import Image

from local_image_search.db import (
    connect,
    count_face_embeddings,
    count_faces,
    ensure_vector_table,
    get_image_id,
    init_db,
    upsert_faces_for_image,
    upsert_indexed_image,
)
from local_image_search.face_detection import FaceDetector
from local_image_search.index_service import index_roots
from local_image_search.models import FaceBox
from local_image_search.stub_embedder import StubEmbedder


class CountingFaceDetector(FaceDetector):
    name = "test-face-detector"

    def __init__(self) -> None:
        self.calls = 0

    def detect_faces(self, image_path: Path) -> list[FaceBox]:
        self.calls += 1
        return [
            FaceBox(
                x=1,
                y=2,
                width=3,
                height=4,
                detection_score=0.9,
                embedding=[1.0] + [0.0] * 511,
            )
        ]


def test_index_roots_stores_face_boxes_and_skips_unchanged_faces(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "images.db"
    album_path = tmp_path / "album"
    album_path.mkdir()
    image_path = album_path / "portrait.jpg"
    Image.new("RGB", (24, 24), "white").save(image_path)
    detector = CountingFaceDetector()

    first_result = index_roots(
        db_path,
        [album_path],
        StubEmbedder(),
        face_detector=detector,
    )
    second_result = index_roots(
        db_path,
        [album_path],
        StubEmbedder(),
        face_detector=detector,
    )

    with connect(db_path) as conn:
        init_db(conn)
        face_count = count_faces(conn)
        face_embedding_count = count_face_embeddings(conn)

    assert first_result.indexed == 1
    assert first_result.faces_indexed == 1
    assert second_result.indexed == 0
    assert second_result.skipped == 1
    assert second_result.faces_indexed == 0
    assert detector.calls == 1
    assert face_count == 1
    assert face_embedding_count == 1


def test_index_roots_backfills_missing_face_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "images.db"
    album_path = tmp_path / "album"
    album_path.mkdir()
    image_path = album_path / "portrait.jpg"
    Image.new("RGB", (24, 24), "white").save(image_path)
    image_file = _image_file(image_path)
    embedder = StubEmbedder()
    detector = CountingFaceDetector()

    with connect(db_path) as conn:
        init_db(conn)
        ensure_vector_table(conn)
        image_id = upsert_indexed_image(
            conn,
            image_file,
            embedder.name,
            embedder.embed_image(image_path),
            None,
        )
        upsert_faces_for_image(
            conn,
            image_id,
            detector.name,
            [FaceBox(x=1, y=2, width=3, height=4, detection_score=0.9)],
        )
        conn.commit()

    result = index_roots(
        db_path,
        [album_path],
        embedder,
        face_detector=detector,
    )

    with connect(db_path) as conn:
        init_db(conn)
        image_id = get_image_id(conn, image_path)
        face_count = count_faces(conn)
        face_embedding_count = count_face_embeddings(conn)
        face_embedding_model = conn.execute(
            "SELECT embedding_model FROM faces WHERE image_id = ?",
            (image_id,),
        ).fetchone()["embedding_model"]

    assert result.indexed == 0
    assert result.skipped == 1
    assert result.faces_indexed == 1
    assert detector.calls == 1
    assert face_count == 1
    assert face_embedding_count == 1
    assert face_embedding_model == detector.embedding_model


def _image_file(image_path: Path):
    stat = image_path.stat()
    from local_image_search.models import ImageFile

    return ImageFile(
        path=image_path,
        file_name=image_path.name,
        file_size=stat.st_size,
        created_at=None,
        modified_at=stat.st_mtime,
    )
