from __future__ import annotations

from pathlib import Path

from PIL import Image

from local_image_search.thumbnails import ensure_thumbnail


def test_ensure_thumbnail_writes_cached_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    thumbnail_dir = tmp_path / "thumbs"
    Image.new("RGBA", (640, 480), (255, 0, 0, 128)).save(image_path)

    thumbnail_path = ensure_thumbnail(image_path, thumbnail_dir=thumbnail_dir, size=128)
    second_path = ensure_thumbnail(image_path, thumbnail_dir=thumbnail_dir, size=128)

    assert thumbnail_path == second_path
    assert thumbnail_path.is_absolute()
    assert thumbnail_path.exists()
    assert thumbnail_path.suffix == ".jpg"

    with Image.open(thumbnail_path) as thumbnail:
        assert thumbnail.mode == "RGB"
        assert max(thumbnail.size) <= 128
