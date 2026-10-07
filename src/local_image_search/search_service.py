from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
import time
from pathlib import Path

from local_image_search.db import (
    connect_readonly,
    count_face_embeddings,
    count_faces,
    count_images,
    count_searchable_images,
    get_primary_face_for_image,
    get_vector_dimensions,
    search_indexed_images_page,
    search_similar_faces_page,
)
from local_image_search.embedder import Embedder
from local_image_search.metrics import memory_status
from local_image_search.models import (
    FaceSearchResult,
    IndexedFace,
    SearchCursor,
    SearchPage,
    SearchResult,
)

API_VERSION = 4


class SearchService:
    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.started_at = time.time()

    def status(self) -> dict:
        try:
            with connect_readonly(self.db_path) as conn:
                total = count_images(conn)
                faces = count_faces(conn)
                face_embeddings = count_face_embeddings(conn)
                searchable = count_searchable_images(conn, self.embedder.name)
                vector_dimensions = get_vector_dimensions(conn)
        except (FileNotFoundError, sqlite3.OperationalError):
            total = 0
            faces = 0
            face_embeddings = 0
            searchable = 0
            vector_dimensions = None
        return {
            "apiVersion": API_VERSION,
            "database": str(self.db_path),
            "embedder": self.embedder.name,
            "embeddingDimensions": self.embedder.dimensions,
            "modelLoaded": getattr(self.embedder, "loaded", True),
            "indexEmbeddingDimensions": vector_dimensions,
            "indexedImages": total,
            "indexedFaces": faces,
            "indexedFaceEmbeddings": face_embeddings,
            "memory": memory_status(),
            "searchableImages": searchable,
            "uptimeSeconds": round(time.time() - self.started_at, 3),
        }

    def search(self, query: str, limit: int, cursor: str | None = None) -> dict:
        started = time.perf_counter()
        search_cursor = _decode_cursor(cursor)
        with connect_readonly(self.db_path) as conn:
            page = search_indexed_images_page(
                conn,
                self.embedder.embed_text(query),
                self.embedder.name,
                limit,
                cursor=search_cursor,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "limit": limit,
            "elapsedMs": round(elapsed_ms, 3),
            **_serialize_search_page(page, _serialize_results),
        }

    def similar(self, image_path: Path, limit: int, cursor: str | None = None) -> dict:
        image_path = image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        started = time.perf_counter()
        search_cursor = _decode_cursor(cursor)
        with connect_readonly(self.db_path) as conn:
            page = search_indexed_images_page(
                conn,
                self.embedder.embed_image(image_path),
                self.embedder.name,
                limit,
                exclude_path=image_path,
                cursor=search_cursor,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "path": str(image_path),
            "limit": limit,
            "elapsedMs": round(elapsed_ms, 3),
            **_serialize_search_page(page, _serialize_results),
        }

    def similar_face(self, face_id: int, limit: int, cursor: str | None = None) -> dict:
        started = time.perf_counter()
        search_cursor = _decode_cursor(cursor)
        with connect_readonly(self.db_path) as conn:
            page = search_similar_faces_page(conn, face_id, limit, cursor=search_cursor)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "faceId": face_id,
            "limit": limit,
            "elapsedMs": round(elapsed_ms, 3),
            **_serialize_search_page(page, _serialize_face_results),
        }

    def primary_face(self, image_id: int) -> dict:
        with connect_readonly(self.db_path) as conn:
            face = get_primary_face_for_image(conn, image_id)
        if face is None:
            raise ValueError(f"No indexed face embedding was found for image: {image_id}")
        return _serialize_face(face)


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


def _serialize_search_page(page: SearchPage, serializer) -> dict:
    return {
        "results": serializer(page.results),
        "hasMore": page.has_more,
        "nextCursor": _encode_cursor(page.next_cursor),
    }


def _encode_cursor(cursor: SearchCursor | None) -> str | None:
    if cursor is None:
        return None
    payload = json.dumps(
        {
            "distance": cursor.distance,
            "rowid": cursor.rowid,
            "seenImageIds": list(cursor.seen_image_ids),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> SearchCursor | None:
    if value is None:
        return None
    try:
        padded_value = value + ("=" * (-len(value) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded_value).decode("utf-8"))
        distance = payload["distance"]
        rowid = payload["rowid"]
        seen_image_ids = payload.get("seenImageIds", [])
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise ValueError("Invalid search cursor") from exc
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
        or isinstance(rowid, bool)
        or not isinstance(rowid, int)
        or rowid < 1
        or not isinstance(seen_image_ids, list)
        or any(
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id < 1
            for image_id in seen_image_ids
        )
    ):
        raise ValueError("Invalid search cursor")
    return SearchCursor(
        distance=float(distance),
        rowid=rowid,
        seen_image_ids=tuple(sorted(set(seen_image_ids))),
    )


def _serialize_face_results(results: list[FaceSearchResult]) -> list[dict]:
    return [
        {
            "faceId": result.face.id,
            "imageId": result.face.image.id,
            "path": str(result.face.image.path),
            "fileName": result.face.image.file_name,
            "score": round(result.score, 6),
            "faceEmbeddingModel": result.face.embedding_model,
            "thumbnailPath": (
                str(result.face.image.thumbnail_path)
                if result.face.image.thumbnail_path
                else None
            ),
            "box": {
                "x": round(result.face.box.x, 3),
                "y": round(result.face.box.y, 3),
                "width": round(result.face.box.width, 3),
                "height": round(result.face.box.height, 3),
                "detectionScore": (
                    None
                    if result.face.box.detection_score is None
                    else round(result.face.box.detection_score, 6)
                ),
            },
        }
        for result in results
    ]


def _serialize_face(face: IndexedFace) -> dict:
    return {
        "faceId": face.id,
        "imageId": face.image.id,
        "path": str(face.image.path),
        "fileName": face.image.file_name,
        "faceEmbeddingModel": face.embedding_model,
        "thumbnailPath": str(face.image.thumbnail_path) if face.image.thumbnail_path else None,
        "box": {
            "x": round(face.box.x, 3),
            "y": round(face.box.y, 3),
            "width": round(face.box.width, 3),
            "height": round(face.box.height, 3),
            "detectionScore": (
                None
                if face.box.detection_score is None
                else round(face.box.detection_score, 6)
            ),
        },
    }
