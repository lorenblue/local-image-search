from __future__ import annotations

from pathlib import Path

from local_image_search.clip import ClipEmbedder
from local_image_search.search_service import SearchService


def create_app(db_path: Path, clip_embedder: ClipEmbedder):
    try:
        from fastapi import FastAPI, HTTPException, Query
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

    @app.get("/similar")
    def similar(
        path: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> dict:
        try:
            return service.similar(Path(path), limit)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
