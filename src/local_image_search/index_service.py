from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from local_image_search.clip import ClipEmbedder
from local_image_search.db import (
    connect,
    delete_missing_paths,
    ensure_vector_table,
    get_image_id,
    get_thumbnail_path,
    init_db,
    needs_face_indexing,
    needs_indexing,
    update_thumbnail_path,
    upsert_faces_for_image,
    upsert_indexed_image,
)
from local_image_search.face_detection import FaceDetector
from local_image_search.scanner import scan_images
from local_image_search.thumbnails import ensure_thumbnail


@dataclass
class IndexProgress:
    roots: list[str] = field(default_factory=list)
    running: bool = False
    total: int = 0
    processed: int = 0
    indexed: int = 0
    faces_indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    last_file: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "roots": self.roots,
            "running": self.running,
            "total": self.total,
            "processed": self.processed,
            "indexed": self.indexed,
            "facesIndexed": self.faces_indexed,
            "skipped": self.skipped,
            "deleted": self.deleted,
            "lastFile": self.last_file,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "error": self.error,
        }

    def copy(self) -> IndexProgress:
        return IndexProgress(
            roots=list(self.roots),
            running=self.running,
            total=self.total,
            processed=self.processed,
            indexed=self.indexed,
            faces_indexed=self.faces_indexed,
            skipped=self.skipped,
            deleted=self.deleted,
            last_file=self.last_file,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
        )


ProgressCallback = Callable[[IndexProgress], None]


def index_roots(
    db_path: Path,
    roots: Iterable[Path],
    clip_embedder: ClipEmbedder,
    face_detector: FaceDetector | None = None,
    on_progress: ProgressCallback | None = None,
) -> IndexProgress:
    roots = list(roots)
    thumbnail_dir = _thumbnail_dir_for_db(db_path)
    progress = IndexProgress(
        roots=[str(root) for root in roots],
        running=True,
        started_at=time.time(),
    )
    _notify(on_progress, progress)

    try:
        images = scan_images(roots)
        progress.total = len(images)
        _notify(on_progress, progress)

        with connect(db_path) as conn:
            init_db(conn)
            ensure_vector_table(conn)
            for image in images:
                progress.processed += 1
                progress.last_file = image.file_name
                clip_stale = needs_indexing(conn, image, clip_embedder.name)

                if clip_stale:
                    embedding = clip_embedder.embed_image(image.path)
                    thumbnail_path = ensure_thumbnail(image.path, thumbnail_dir=thumbnail_dir)
                    image_id = upsert_indexed_image(
                        conn,
                        image,
                        clip_embedder.name,
                        embedding,
                        thumbnail_path,
                    )
                    progress.indexed += 1
                else:
                    image_id = get_image_id(conn, image.path)
                    progress.skipped += 1
                    _ensure_stored_thumbnail(conn, image.path, thumbnail_dir)

                if face_detector is not None and needs_face_indexing(
                    conn,
                    image.path,
                    face_detector.name,
                    face_detector.embedding_model,
                ):
                    progress.faces_indexed += _detect_and_store_faces(
                        conn,
                        image_id,
                        image.path,
                        face_detector,
                    )
                conn.commit()
                _notify(on_progress, progress)

            progress.deleted = delete_missing_paths(
                conn,
                [image.path for image in images],
                roots,
            )
            conn.commit()
    except Exception as exc:
        progress.error = str(exc)
        raise
    finally:
        progress.running = False
        progress.finished_at = time.time()
        _notify(on_progress, progress)

    return progress


class BackgroundIndexService:
    def __init__(
        self,
        db_path: Path,
        clip_embedder: ClipEmbedder,
        face_detector: FaceDetector | None = None,
    ) -> None:
        self.db_path = db_path
        self.clip_embedder = clip_embedder
        self.face_detector = face_detector
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._progress = IndexProgress()

    def status(self) -> dict:
        with self._lock:
            return self._progress.copy().to_dict()

    def start(self, roots: Iterable[Path]) -> dict:
        roots = list(roots)
        if not roots:
            raise ValueError("At least one folder is required")

        with self._lock:
            if self._progress.running:
                return self._progress.copy().to_dict()

            self._progress = IndexProgress(
                roots=[str(root) for root in roots],
                running=True,
                started_at=time.time(),
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(roots,),
                name="local-image-search-indexer",
                daemon=True,
            )
            self._thread.start()
            return self._progress.copy().to_dict()

    def _run(self, roots: list[Path]) -> None:
        try:
            index_roots(
                self.db_path,
                roots,
                self.clip_embedder,
                face_detector=self.face_detector,
                on_progress=self._set_progress,
            )
        except Exception as exc:
            with self._lock:
                self._progress.running = False
                self._progress.finished_at = time.time()
                self._progress.error = str(exc)

    def _set_progress(self, progress: IndexProgress) -> None:
        with self._lock:
            self._progress = progress.copy()


def _thumbnail_dir_for_db(db_path: Path) -> Path:
    return db_path.expanduser().resolve().parent / "thumbnails"


def _ensure_stored_thumbnail(
    conn: sqlite3.Connection,
    image_path: Path,
    thumbnail_dir: Path,
) -> None:
    existing = get_thumbnail_path(conn, image_path)
    if existing is not None and existing.exists():
        return
    thumbnail_path = ensure_thumbnail(image_path, thumbnail_dir=thumbnail_dir)
    if thumbnail_path is None:
        return
    update_thumbnail_path(conn, image_path, thumbnail_path)


def _detect_and_store_faces(
    conn: sqlite3.Connection,
    image_id: int,
    image_path: Path,
    face_detector: FaceDetector,
) -> int:
    faces = face_detector.detect_faces(image_path)
    upsert_faces_for_image(
        conn,
        image_id,
        face_detector.name,
        faces,
        embedding_model=face_detector.embedding_model,
    )
    return len(faces)


def _notify(callback: ProgressCallback | None, progress: IndexProgress) -> None:
    if callback is not None:
        callback(progress.copy())
