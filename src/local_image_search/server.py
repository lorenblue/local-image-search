from __future__ import annotations

import time
from pathlib import Path

from local_image_search.clip import ClipEmbedder
from local_image_search.db import (
    connect_readonly,
    count_images,
    count_searchable_images,
    search_indexed_images,
)
from local_image_search.metrics import memory_status


class SearchService:
    def __init__(self, db_path: Path, clip_embedder: ClipEmbedder) -> None:
        self.db_path = db_path
        self.clip_embedder = clip_embedder
        self.started_at = time.time()
        if clip_embedder.dimensions != 512:
            raise ValueError(
                "sqlite-vec CLIP search expects "
                f"512-dimensional embeddings, got {clip_embedder.dimensions}"
            )

    def status(self) -> dict:
        with connect_readonly(self.db_path) as conn:
            total = count_images(conn)
            searchable = count_searchable_images(conn, self.clip_embedder.name)
        return {
            "database": str(self.db_path),
            "clipEmbedder": self.clip_embedder.name,
            "indexedImages": total,
            "memory": memory_status(),
            "searchableImages": searchable,
            "uptimeSeconds": round(time.time() - self.started_at, 3),
        }

    def search(self, query: str, limit: int) -> dict:
        started = time.perf_counter()
        with connect_readonly(self.db_path) as conn:
            results = search_indexed_images(
                conn,
                self.clip_embedder.embed_text(query),
                self.clip_embedder.name,
                limit,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "limit": limit,
            "elapsedMs": round(elapsed_ms, 3),
            "results": [
                {
                    "id": result.image.id,
                    "path": str(result.image.path),
                    "fileName": result.image.file_name,
                    "score": round(result.score, 6),
                    "embeddingModel": result.image.embedding_model,
                    "thumbnailPath": (
                        str(result.image.thumbnail_path)
                        if result.image.thumbnail_path
                        else None
                    ),
                }
                for result in results
            ],
        }


def create_app(db_path: Path, clip_embedder: ClipEmbedder):
    try:
        from fastapi import FastAPI, Query
        from scalar_fastapi import get_scalar_api_reference
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    service = SearchService(db_path, clip_embedder)
    app = FastAPI(title="Local Image Search API")

    @app.get("/scalar", include_in_schema=False)
    def scalar_reference():
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title="Local Image Search API",
            telemetry=False,
        )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/status")
    def status() -> dict:
        return service.status()

    @app.get("/search")
    def search(
        q: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> dict:
        return service.search(q.strip(), limit)

    app.state.search_service = service
    return app


def run_server(
    db_path: Path,
    clip_embedder: ClipEmbedder,
    host: str,
    port: int,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    app = create_app(db_path, clip_embedder)
    print(f"serving search API on http://{host}:{port}")
    print(f"using CLIP search with {clip_embedder.name}")
    uvicorn.run(app, host=host, port=port)
