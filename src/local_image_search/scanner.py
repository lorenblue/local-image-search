from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from local_image_search.config import SUPPORTED_EXTENSIONS
from local_image_search.models import ImageFile


def scan_images(roots: Iterable[Path]) -> list[ImageFile]:
    images: list[ImageFile] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]

        for path in candidates:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            images.append(
                ImageFile(
                    path=path,
                    file_name=path.name,
                    file_size=stat.st_size,
                    created_at=getattr(stat, "st_birthtime", None),
                    modified_at=stat.st_mtime,
                )
            )

    return sorted(images, key=lambda image: str(image.path))
