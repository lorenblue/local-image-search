from __future__ import annotations

import math

import numpy as np

from local_image_search.embeddings import Embedder
from local_image_search.models import IndexedImage, SearchResult


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def search_images(
    query: str,
    images: list[IndexedImage],
    embedder: Embedder,
    limit: int,
) -> list[SearchResult]:
    query_embedding = embedder.embed(query)
    results = [
        SearchResult(image=image, score=cosine_similarity(query_embedding, image.embedding))
        for image in images
        if image.embedding_model == embedder.name
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]


class SearchIndex:
    def __init__(self, images: list[IndexedImage], embedder: Embedder) -> None:
        self.embedder = embedder
        self.images = [image for image in images if image.embedding_model == embedder.name]
        self._matrix = self._build_matrix(self.images)

    @property
    def size(self) -> int:
        return len(self.images)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if self._matrix.size == 0:
            return []

        query_embedding = np.array(self.embedder.embed(query), dtype=np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return []
        query_embedding = query_embedding / norm

        scores = self._matrix @ query_embedding
        limit = max(0, min(limit, len(self.images)))
        if limit == 0:
            return []

        top_indices = np.argpartition(scores, -limit)[-limit:]
        ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            SearchResult(image=self.images[int(index)], score=float(scores[int(index)]))
            for index in ranked_indices
        ]

    @staticmethod
    def _build_matrix(images: list[IndexedImage]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        matrix = np.array([image.embedding for image in images], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
