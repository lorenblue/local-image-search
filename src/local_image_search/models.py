from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageFile:
    path: Path
    file_name: str
    file_size: int
    created_at: float | None
    modified_at: float


@dataclass(frozen=True)
class IndexedImage:
    id: int
    path: Path
    file_name: str
    file_size: int
    created_at: float | None
    modified_at: float
    caption: str
    caption_model: str
    embedding_model: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    image: IndexedImage
    score: float
