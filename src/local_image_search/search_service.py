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
from local_image_search.models import SearchResult


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
            "results": _serialize_results(results),
        }

    def similar(self, image_path: Path, limit: int) -> dict:
        image_path = image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        started = time.perf_counter()
        with connect_readonly(self.db_path) as conn:
            results = search_indexed_images(
                conn,
                self.clip_embedder.embed_image(image_path),
                self.clip_embedder.name,
                limit,
                exclude_path=image_path,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "path": str(image_path),
            "limit": limit,
            "elapsedMs": round(elapsed_ms, 3),
            "results": _serialize_results(results),
        }


def _serialize_results(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "id": result.image.id,
            "path": str(result.image.path),
            "fileName": result.image.file_name,
            "score": round(result.score, 6),
            "embeddingModel": result.image.embedding_model,
            "thumbnailPath": (
                str(result.image.thumbnail_path) if result.image.thumbnail_path else None
            ),
        }
        for result in results
    ]
