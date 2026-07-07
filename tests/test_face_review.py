from __future__ import annotations

from pathlib import Path

from PIL import Image

from local_image_search.db import (
    connect,
    ensure_vector_table,
    init_db,
    upsert_faces_for_image,
    upsert_indexed_image,
)
from local_image_search.face_review import write_face_review
from local_image_search.models import FaceBox, ImageFile


def test_write_face_review_draws_stored_detected_boxes(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)

    _insert_face(
        db_path,
        image_path,
        FaceBox(x=10, y=20, width=30, height=40, detection_score=0.9),
    )

    output_path = write_face_review(
        db_path,
        [tmp_path],
        tmp_path / "faces.html",
        limit=10,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Face Review" in html
    assert "portrait.jpg" in html
    assert "1 stored faces" in html
    assert "test-face" in html
    assert "#1" in html
    assert "left:10.000%;top:10.000%;width:30.000%;height:20.000%" in html


def test_write_face_review_skips_unreadable_indexed_images(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    image_path = tmp_path / "not-really-an-image.jpg"
    image_path.write_bytes(b"not an image")

    _insert_face(
        db_path,
        image_path,
        FaceBox(x=10, y=20, width=30, height=40, detection_score=0.9),
    )

    output_path = write_face_review(
        db_path,
        [],
        tmp_path / "faces.html",
        limit=10,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "0 images" in html
    assert "not-really-an-image.jpg" not in html


def test_write_face_review_filters_to_requested_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    included_dir = tmp_path / "included"
    excluded_dir = tmp_path / "excluded"
    included_dir.mkdir()
    excluded_dir.mkdir()
    included_path = included_dir / "included.jpg"
    excluded_path = excluded_dir / "excluded.jpg"
    Image.new("RGB", (100, 100), "white").save(included_path)
    Image.new("RGB", (100, 100), "white").save(excluded_path)

    _insert_face(db_path, included_path, FaceBox(1, 2, 3, 4, 0.9))
    _insert_face(db_path, excluded_path, FaceBox(5, 6, 7, 8, 0.8))

    output_path = write_face_review(
        db_path,
        [included_dir],
        tmp_path / "faces.html",
        limit=10,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "included.jpg" in html
    assert "excluded.jpg" not in html


def _insert_face(db_path: Path, image_path: Path, face: FaceBox) -> None:
    image = ImageFile(
        path=image_path,
        file_name=image_path.name,
        file_size=image_path.stat().st_size,
        created_at=None,
        modified_at=image_path.stat().st_mtime,
    )
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
        upsert_faces_for_image(conn, image_id, "test-face", [face])
        conn.commit()
