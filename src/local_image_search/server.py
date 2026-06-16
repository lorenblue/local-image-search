from __future__ import annotations

import time
from pathlib import Path

from local_image_search.db import connect, count_images, init_db, list_indexed_images
from local_image_search.embeddings import Embedder
from local_image_search.metrics import memory_status
from local_image_search.search import SearchIndex


class SearchService:
    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.started_at = time.time()
        self.index = self._load_index()

    def reload(self) -> None:
        self.index = self._load_index()

    def status(self) -> dict:
        with connect(self.db_path) as conn:
            init_db(conn)
            total = count_images(conn)
        return {
            "database": str(self.db_path),
            "embedder": self.embedder.name,
            "indexedImages": total,
            "memory": memory_status(),
            "searchableImages": self.index.size,
            "uptimeSeconds": round(time.time() - self.started_at, 3),
        }

    def search(self, query: str, limit: int) -> dict:
        started = time.perf_counter()
        results = self.index.search(query, limit)
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
                    "caption": result.image.caption,
                    "captionModel": result.image.caption_model,
                    "embeddingModel": result.image.embedding_model,
                }
                for result in results
            ],
        }

    def _load_index(self) -> SearchIndex:
        with connect(self.db_path) as conn:
            init_db(conn)
            images = list_indexed_images(conn)
        return SearchIndex(images, self.embedder)


def create_app(db_path: Path, embedder: Embedder):
    try:
        from fastapi import FastAPI, Query
        from scalar_fastapi import get_scalar_api_reference
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    service = SearchService(db_path, embedder)
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

    @app.post("/reload")
    def reload_index() -> dict:
        service.reload()
        return {"ok": True, "searchableImages": service.index.size}

    app.state.search_service = service
    return app


def run_server(
    db_path: Path,
    embedder: Embedder,
    host: str,
    port: int,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    app = create_app(db_path, embedder)
    service = app.state.search_service
    print(f"serving search API on http://{host}:{port}")
    print(f"loaded {service.index.size} searchable images with {embedder.name}")
    uvicorn.run(app, host=host, port=port)
