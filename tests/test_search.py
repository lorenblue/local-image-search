from __future__ import annotations

from local_image_search.embeddings import StubEmbedder
from local_image_search.search import cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_stub_embedder_makes_related_text_closer() -> None:
    embedder = StubEmbedder()
    query = embedder.embed("girl in car")
    related = embedder.embed("woman sitting in car")
    unrelated = embedder.embed("dog on beach")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)
