from __future__ import annotations

import math

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
