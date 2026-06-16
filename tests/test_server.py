from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from local_image_search.db import connect, init_db, upsert_indexed_image
from local_image_search.embeddings import StubEmbedder
from local_image_search.metrics import memory_status
from local_image_search.models import ImageFile
from local_image_search.server import create_app


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
        upsert_indexed_image(
            conn,
            image,
            caption,
            caption_model="stub-captioner-v1",
            embedding_model=embedder.name,
            embedding=embedder.embed(caption),
            thumbnail_path=tmp_path / "thumb.jpg",
        )
        conn.commit()

    client = TestClient(create_app(db_path, embedder))

    assert client.get("/health").json() == {"ok": True}
    status = client.get("/status").json()
    assert status["searchableImages"] == 1
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
