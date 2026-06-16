from __future__ import annotations

from pathlib import Path

from local_image_search.scanner import scan_images


def test_scan_images_filters_supported_extensions(tmp_path: Path) -> None:
    image = tmp_path / "person-wearing-glasses.jpg"
    image.write_bytes(b"fake image bytes")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("not an image")

    results = scan_images([tmp_path])

    assert [result.path for result in results] == [image]
