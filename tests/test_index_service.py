from __future__ import annotations

from pathlib import Path

from PIL import Image

from local_image_search.clip import StubClipEmbedder
from local_image_search.db import connect, count_faces, init_db
from local_image_search.face_detection import FaceDetector
from local_image_search.index_service import index_roots
from local_image_search.models import FaceBox


class CountingFaceDetector(FaceDetector):
    name = "test-face-detector"

    def __init__(self) -> None:
        self.calls = 0

    def detect_faces(self, image_path: Path) -> list[FaceBox]:
        self.calls += 1
        return [FaceBox(x=1, y=2, width=3, height=4, detection_score=0.9)]


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
        StubClipEmbedder(),
        face_detector=detector,
    )
    second_result = index_roots(
        db_path,
        [album_path],
        StubClipEmbedder(),
        face_detector=detector,
    )

    with connect(db_path) as conn:
        init_db(conn)
        face_count = count_faces(conn)

    assert first_result.indexed == 1
    assert first_result.faces_indexed == 1
    assert second_result.indexed == 0
    assert second_result.skipped == 1
    assert second_result.faces_indexed == 0
    assert detector.calls == 1
    assert face_count == 1
