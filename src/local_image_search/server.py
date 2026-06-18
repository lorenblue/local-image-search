from __future__ import annotations

import time
from pathlib import Path

from local_image_search.clip import ClipEmbedder
from local_image_search.db import (
    connect_readonly,
    count_clip_searchable_images,
    count_images,
    count_searchable_images,
    search_clip_images,
    search_indexed_images,
)
from local_image_search.embeddings import Embedder
from local_image_search.metrics import memory_status


class SearchService:
    def __init__(
        self,
        db_path: Path,
        embedder: Embedder,
        clip_embedder: ClipEmbedder | None = None,
    ) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.clip_embedder = clip_embedder
        self.started_at = time.time()
        if embedder.dimensions != 384:
            raise ValueError(
                f"sqlite-vec search expects 384-dimensional embeddings, got {embedder.dimensions}"
            )
        if clip_embedder is not None and clip_embedder.dimensions != 512:
            raise ValueError(
                "sqlite-vec CLIP search expects "
                f"512-dimensional embeddings, got {clip_embedder.dimensions}"
            )

    def status(self) -> dict:
        with connect_readonly(self.db_path) as conn:
            total = count_images(conn)
            searchable = count_searchable_images(conn, self.embedder.name)
            clip_searchable = (
                count_clip_searchable_images(conn, self.clip_embedder.name)
                if self.clip_embedder
                else 0
            )
        return {
            "database": str(self.db_path),
            "embedder": self.embedder.name,
            "clipEmbedder": self.clip_embedder.name if self.clip_embedder else None,
            "indexedImages": total,
            "memory": memory_status(),
            "searchableImages": searchable,
            "clipSearchableImages": clip_searchable,
            "uptimeSeconds": round(time.time() - self.started_at, 3),
        }

    def search(self, query: str, limit: int, mode: str) -> dict:
        started = time.perf_counter()
        with connect_readonly(self.db_path) as conn:
            if mode == "caption":
                results = search_indexed_images(
                    conn,
                    self.embedder.embed(query),
                    self.embedder.name,
                    limit,
                )
            elif mode == "clip":
                if self.clip_embedder is None:
                    raise ValueError("CLIP search is not enabled on this server")
                results = search_clip_images(
                    conn,
                    self.clip_embedder.embed_text(query),
                    self.clip_embedder.name,
                    limit,
                )
            else:
                raise ValueError(f"Unknown search mode: {mode}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "limit": limit,
            "mode": mode,
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
                    "thumbnailPath": (
                        str(result.image.thumbnail_path)
                        if result.image.thumbnail_path
                        else None
                    ),
                }
                for result in results
            ],
        }


def create_app(
    db_path: Path,
    embedder: Embedder,
    clip_embedder: ClipEmbedder | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from scalar_fastapi import get_scalar_api_reference
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    service = SearchService(db_path, embedder, clip_embedder)
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
        mode: str = Query(default="caption", pattern="^(caption|clip)$"),
    ) -> dict:
        try:
            return service.search(q.strip(), limit, mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.state.search_service = service
    return app


def run_server(
    db_path: Path,
    embedder: Embedder,
    clip_embedder: ClipEmbedder | None,
    host: str,
    port: int,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    app = create_app(db_path, embedder, clip_embedder)
    print(f"serving search API on http://{host}:{port}")
    print(f"using sqlite-vec search with {embedder.name}")
    if clip_embedder:
        print(f"using CLIP search with {clip_embedder.name}")
    uvicorn.run(app, host=host, port=port)
