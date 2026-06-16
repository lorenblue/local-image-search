from __future__ import annotations

from pathlib import Path

from local_image_search.embeddings import StubEmbedder
from local_image_search.models import IndexedImage
from local_image_search.search import SearchIndex, cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_stub_embedder_makes_related_text_closer() -> None:
    embedder = StubEmbedder()
    query = embedder.embed("girl in car")
    related = embedder.embed("woman sitting in car")
    unrelated = embedder.embed("dog on beach")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_search_index_ranks_matching_caption_first() -> None:
    embedder = StubEmbedder()
    images = [
        IndexedImage(
            id=1,
            path=Path("dog.jpg"),
            file_name="dog.jpg",
            file_size=10,
            created_at=None,
            modified_at=1,
            caption="A dog running on a beach",
            caption_model="stub",
            embedding_model=embedder.name,
            embedding=embedder.embed("A dog running on a beach"),
        ),
        IndexedImage(
            id=2,
            path=Path("car.jpg"),
            file_name="car.jpg",
            file_size=10,
            created_at=None,
            modified_at=1,
            caption="A woman sitting in a car",
            caption_model="stub",
            embedding_model=embedder.name,
            embedding=embedder.embed("A woman sitting in a car"),
        ),
    ]

    results = SearchIndex(images, embedder).search("girl in car", limit=1)

    assert results[0].image.file_name == "car.jpg"
