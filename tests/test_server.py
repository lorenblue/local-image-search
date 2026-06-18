from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from local_image_search.clip import StubClipEmbedder
from local_image_search.db import connect, ensure_vector_table, init_db, upsert_indexed_image
from local_image_search.metrics import memory_status
from local_image_search.models import ImageFile
from local_image_search.server import SearchService, create_app


def test_api_search_returns_ranked_clip_results(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
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
        _insert_indexed_image(conn, image, clip_embedder, thumbnail_path=tmp_path / "thumb.jpg")
        conn.commit()

    client = TestClient(create_app(db_path, clip_embedder))

    assert client.get("/health").json() == {"ok": True}
    status = client.get("/status").json()
    assert status["searchableImages"] == 1
    assert status["clipEmbedder"] == clip_embedder.name
    assert "memory" in status
    assert status["memory"]["currentMb"] > 0
    assert status["memory"]["peakMb"] > 0
    assert client.get("/scalar").status_code == 200

    response = client.get("/search", params={"q": "red sports car", "limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["fileName"] == "red-sports-car.jpg"
    assert body["results"][0]["thumbnailPath"] == str(tmp_path / "thumb.jpg")


def test_api_similar_returns_ranked_results_and_excludes_source(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    clip_embedder = StubClipEmbedder()
    source_path = tmp_path / "red-sports-car.jpg"
    similar_path = tmp_path / "red-sports-truck.jpg"
    other_path = tmp_path / "beach-sunset.jpg"
    for path in [source_path, similar_path, other_path]:
        path.write_bytes(b"test image placeholder")

    images = [
        ImageFile(
            path=source_path,
            file_name=source_path.name,
            file_size=10,
            created_at=None,
            modified_at=1,
        ),
        ImageFile(
            path=similar_path,
            file_name=similar_path.name,
            file_size=10,
            created_at=None,
            modified_at=1,
        ),
        ImageFile(
            path=other_path,
            file_name=other_path.name,
            file_size=10,
            created_at=None,
            modified_at=1,
        ),
    ]

    with connect(db_path) as conn:
        init_db(conn)
        for image in images:
            _insert_indexed_image(conn, image, clip_embedder, thumbnail_path=None)
        conn.commit()

    client = TestClient(create_app(db_path, clip_embedder))

    response = client.get("/similar", params={"path": str(source_path), "limit": 2})

    assert response.status_code == 200
    body = response.json()
    result_names = [result["fileName"] for result in body["results"]]
    assert body["path"] == str(source_path.resolve())
    assert source_path.name not in result_names
    assert result_names[0] == similar_path.name


def test_api_similar_returns_404_for_missing_reference_image(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    clip_embedder = StubClipEmbedder()
    with connect(db_path) as conn:
        init_db(conn)
        conn.commit()

    client = TestClient(create_app(db_path, clip_embedder))

    response = client.get("/similar", params={"path": str(tmp_path / "missing.jpg")})

    assert response.status_code == 404


def test_memory_status_reports_current_and_peak_memory() -> None:
    memory = memory_status()

    assert memory["currentMb"] > 0
    assert memory["peakMb"] > 0


def test_search_service_finds_newly_committed_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    clip_embedder = StubClipEmbedder()

    with connect(db_path) as conn:
        init_db(conn)

    service = SearchService(db_path, clip_embedder)
    assert service.status()["searchableImages"] == 0

    image = ImageFile(
        path=tmp_path / "new-car.jpg",
        file_name="new-car.jpg",
        file_size=10,
        created_at=None,
        modified_at=1,
    )
    with connect(db_path) as conn:
        ensure_vector_table(conn)
        _insert_indexed_image(conn, image, clip_embedder, thumbnail_path=None)
        conn.commit()

    results = service.search("new car", limit=1)

    assert service.status()["searchableImages"] == 1
    assert results["results"][0]["fileName"] == "new-car.jpg"


def _insert_indexed_image(
    conn,
    image: ImageFile,
    clip_embedder: StubClipEmbedder,
    thumbnail_path: Path | None,
) -> None:
    ensure_vector_table(conn)
    upsert_indexed_image(
        conn,
        image,
        clip_embedder.name,
        clip_embedder.embed_image(image.path),
        thumbnail_path=thumbnail_path,
    )
