from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from local_image_search.config import DEFAULT_THUMBNAIL_DIR


def thumbnail_path_for_image(
    image_path: Path,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
) -> Path:
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:24]
    return (thumbnail_dir / f"{digest}.jpg").resolve()


def ensure_thumbnail(
    image_path: Path,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    size: int = 320,
) -> Path | None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        pass
    else:
        register_heif_opener()

    thumbnail_path = thumbnail_path_for_image(image_path, thumbnail_dir)
    if thumbnail_path.exists():
        return thumbnail_path

    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((size, size))
            image = _flatten_to_rgb(image)
            image.save(thumbnail_path, format="JPEG", quality=82, optimize=True)
    except (OSError, UnidentifiedImageError):
        return None
    return thumbnail_path


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")
