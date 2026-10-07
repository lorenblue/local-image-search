from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Embedder(ABC):
    name: str
    dimensions: int

    def close(self) -> None:
        """Release backend resources, if any."""
        return None

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @abstractmethod
    def embed_image(self, image_path: Path) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError
