from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from local_image_search.clip import StubClipEmbedder
from local_image_search.db import (
    connect,
    ensure_clip_vector_table,
    ensure_vector_table,
    get_image_id,
    init_db,
    upsert_clip_image_embedding,
    upsert_image_embedding,
    upsert_image_metadata,
    upsert_indexed_image,
)
from local_image_search.embeddings import StubEmbedder
from local_image_search.metrics import memory_status
from local_image_search.models import ImageFile
from local_image_search.server import SearchService, create_app


def test_api_search_returns_ranked_results(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    embedder = StubEmbedder()
    image = ImageFile(
        path=tmp_path / "person-wearing-glasses.jpg",
        file_name="person-wearing-glasses.jpg",
        file_size=10,
        created_at=None,
        modified_at=1,
    )
    caption = "A person wearing glasses standing indoors"

    with connect(db_path) as conn:
        init_db(conn)
        _insert_indexed_image(conn, image, caption, embedder, thumbnail_path=tmp_path / "thumb.jpg")
        conn.commit()

    client = TestClient(create_app(db_path, embedder))

    assert client.get("/health").json() == {"ok": True}
    status = client.get("/status").json()
    assert status["searchableImages"] == 1
    assert status["clipEmbedder"] is None
    assert status["clipSearchableImages"] == 0
    assert "memory" in status
    assert status["memory"]["currentMb"] > 0
    assert status["memory"]["peakMb"] > 0
    assert client.get("/scalar").status_code == 200

    response = client.get("/search", params={"q": "person with glasses", "limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["fileName"] == "person-wearing-glasses.jpg"
    assert body["results"][0]["caption"] == caption
    assert body["results"][0]["thumbnailPath"] == str(tmp_path / "thumb.jpg")


def test_memory_status_reports_current_and_peak_memory() -> None:
    memory = memory_status()

    assert memory["currentMb"] > 0
    assert memory["peakMb"] > 0


def test_search_service_finds_newly_committed_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    embedder = StubEmbedder()

    with connect(db_path) as conn:
        init_db(conn)

    service = SearchService(db_path, embedder)
    assert service.status()["searchableImages"] == 0

    image = ImageFile(
        path=tmp_path / "new-car.jpg",
        file_name="new-car.jpg",
        file_size=10,
        created_at=None,
        modified_at=1,
    )
    caption = "A woman sitting in a car"
    with connect(db_path) as conn:
        _insert_indexed_image(conn, image, caption, embedder, thumbnail_path=None)
        conn.commit()

    results = service.search("girl in car", limit=1, mode="caption")

    assert service.status()["searchableImages"] == 1
    assert results["results"][0]["fileName"] == "new-car.jpg"


def test_api_clip_search_returns_ranked_results(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    embedder = StubEmbedder()
    clip_embedder = StubClipEmbedder()
    image = ImageFile(
        path=tmp_path / "red-sports-car.jpg",
        file_name="red-sports-car.jpg",
        file_size=10,
        created_at=None,
        modified_at=1,
    )

    with connect(db_path) as conn:
        init_db(conn)
        _insert_clip_image(conn, image, clip_embedder, thumbnail_path=None)
        conn.commit()

    client = TestClient(create_app(db_path, embedder, clip_embedder))
    status = client.get("/status").json()
    assert status["clipSearchableImages"] == 1

    response = client.get("/search", params={"q": "red sports car", "mode": "clip", "limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "clip"
    assert body["results"][0]["fileName"] == "red-sports-car.jpg"


def test_api_clip_search_requires_enabled_clip_embedder(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    embedder = StubEmbedder()
    with connect(db_path) as conn:
        init_db(conn)

    client = TestClient(create_app(db_path, embedder))
    response = client.get("/search", params={"q": "red sports car", "mode": "clip"})

    assert response.status_code == 400
    assert response.json()["detail"] == "CLIP search is not enabled on this server"


def _insert_indexed_image(
    conn,
    image: ImageFile,
    caption: str,
    embedder: StubEmbedder,
    thumbnail_path: Path | None,
) -> None:
    embedding = embedder.embed(caption)
    ensure_vector_table(conn)
    upsert_indexed_image(
        conn,
        image,
        caption,
        caption_model="stub-captioner-v1",
        embedding_model=embedder.name,
        thumbnail_path=thumbnail_path,
    )
    image_id = get_image_id(conn, image.path)
    upsert_image_embedding(conn, image_id, embedding)


def _insert_clip_image(
    conn,
    image: ImageFile,
    clip_embedder: StubClipEmbedder,
    thumbnail_path: Path | None,
) -> None:
    ensure_clip_vector_table(conn)
    upsert_image_metadata(conn, image, thumbnail_path)
    image_id = get_image_id(conn, image.path)
    upsert_clip_image_embedding(
        conn,
        image_id,
        clip_embedder.name,
        clip_embedder.embed_image(image.path),
    )
