from __future__ import annotations

from pathlib import Path

APP_NAME = "local-image-search"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "images.db"
DEFAULT_THUMBNAIL_DIR = DEFAULT_DATA_DIR / "thumbnails"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".webp",
}
