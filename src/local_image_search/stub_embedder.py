from __future__ import annotations

import hashlib
import math
from pathlib import Path

from local_image_search.embedder import Embedder


class StubEmbedder(Embedder):
    name = "stub-embedder-v1"
    dimensions = 512

    def embed_image(self, image_path: Path) -> list[float]:
        return self._embed(image_path.stem.replace("-", " ").replace("_", " "))

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        for word in words:
            if not word:
                continue
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
