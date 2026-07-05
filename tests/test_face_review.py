from __future__ import annotations

from pathlib import Path

from PIL import Image

from local_image_search.face_detection import FaceDetector
from local_image_search.face_review import write_face_review
from local_image_search.models import FaceBox


class FakeFaceDetector(FaceDetector):
    name = "fake-face"

    def detect_faces(self, image_path: Path) -> list[FaceBox]:
        return [
            FaceBox(
                x=10,
                y=20,
                width=30,
                height=40,
                detection_score=0.9,
            )
        ]


def test_write_face_review_draws_detected_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)

    output_path = write_face_review(
        [tmp_path],
        FakeFaceDetector(),
        tmp_path / "faces.html",
        limit=10,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Face Review" in html
    assert "portrait.jpg" in html
    assert "1 detected faces" in html
    assert "left:10.000%;top:10.000%;width:30.000%;height:20.000%" in html


def test_write_face_review_skips_unreadable_image_files(tmp_path: Path) -> None:
    image_path = tmp_path / "not-really-an-image.jpg"
    image_path.write_bytes(b"not an image")

    output_path = write_face_review(
        [tmp_path],
        FakeFaceDetector(),
        tmp_path / "faces.html",
        limit=10,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "0 images" in html
    assert "not-really-an-image.jpg" not in html
