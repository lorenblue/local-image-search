from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class StubEmbedder(Embedder):
    name = "stub-embedder-v1"
    dimensions = 384

    def embed(self, text: str) -> list[float]:
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


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence transformer embeddings require: python -m pip install -e '.[ml]'"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]


def make_embedder(name: str) -> Embedder:
    normalized = name.lower().strip()
    if normalized == "stub":
        return StubEmbedder()
    if normalized in {"sentence-transformers", "sentence-transformer", "st"}:
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder: {name}")
