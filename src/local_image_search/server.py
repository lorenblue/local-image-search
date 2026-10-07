from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from local_image_search.embedder import Embedder
from local_image_search.face_detection import FaceDetector
from local_image_search.index_service import BackgroundIndexService
from local_image_search.search_service import SearchService


def create_app(
    db_path: Path,
    embedder: Embedder,
    face_detector: FaceDetector | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from scalar_fastapi import get_scalar_api_reference
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    service = SearchService(db_path, embedder)
    indexer = BackgroundIndexService(db_path, embedder, face_detector)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(indexer.wait)
            embedder.close()

    app = FastAPI(title="Local Image Search API", lifespan=lifespan)

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
        response = service.status()
        response["indexing"] = indexer.status()
        return response

    @app.post("/sync")
    def sync(payload: dict) -> dict:
        try:
            roots = _sync_roots(payload)
            return {"indexing": indexer.start(Path(root) for root in roots)}
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/search")
    def search(
        q: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ) -> dict:
        try:
            return service.search(q.strip(), limit, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/similar")
    def similar(
        path: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ) -> dict:
        try:
            return service.similar(Path(path), limit, cursor=cursor)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/similar-face")
    def similar_face(
        face_id: int = Query(alias="faceId", ge=1),
        limit: int = Query(default=10, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ) -> dict:
        try:
            return service.similar_face(face_id, limit, cursor=cursor)
        except ValueError as exc:
            status_code = 400 if str(exc) == "Invalid search cursor" else 404
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.get("/primary-face")
    def primary_face(
        image_id: int = Query(alias="imageId", ge=1),
    ) -> dict:
        try:
            return service.primary_face(image_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.state.search_service = service
    app.state.index_service = indexer
    return app


def _sync_roots(payload: dict) -> list[str]:
    roots = payload.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("At least one folder is required")
    if not all(isinstance(root, str) and root.strip() for root in roots):
        raise ValueError("Sync roots must be non-empty strings")
    return [root.strip() for root in roots]


def run_server(
    db_path: Path,
    embedder: Embedder,
    face_detector: FaceDetector,
    host: str,
    port: int,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires: python -m pip install -e '.[api]'") from exc

    app = create_app(db_path, embedder, face_detector)
    print(f"serving search API on http://{host}:{port}")
    print(f"using semantic search with {embedder.name}")
    print(f"using face detection with {face_detector.name}")
    uvicorn.run(app, host=host, port=port)
