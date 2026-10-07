from __future__ import annotations

import os
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps

from local_image_search.config import DEFAULT_GEMMA_MODEL_PATH
from local_image_search.embedder import Embedder


class EmbeddingGemma2Embedder(Embedder):
    name = "embeddinggemma-2/text-vision-440m"
    dimensions = 768

    def __init__(self, model_path: Path | None = None) -> None:
        configured_path = (
            model_path or os.environ.get("GEMMA_MODEL_PATH") or DEFAULT_GEMMA_MODEL_PATH
        )
        self._model_path = Path(configured_path).expanduser().resolve()
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Gemma model does not exist: {self._model_path}. "
                "Place embeddinggemma-2-text-vision-440m.litertlm there, "
                "or override with --gemma-model or GEMMA_MODEL_PATH."
            )
        try:
            from litert_lm import Backend, Content, EmbeddingEngine
        except ImportError as exc:
            raise RuntimeError(
                "Gemma search requires: python -m pip install -e ."
            ) from exc

        self._backend = Backend.GPU()
        self._content = Content
        self._engine_class = EmbeddingEngine
        self._engine = None
        self._lock = threading.Lock()
        self._closed = False
        self.loaded = False

    def _load(self):
        # Callers serialize engine creation, inference, and disposal with _lock.
        if self._closed:
            raise RuntimeError("Gemma embedder has been closed")
        if self._engine is None:
            self._engine = self._engine_class(
                str(self._model_path),
                backend=self._backend,
                vision_backend=self._backend,
            )
            self.loaded = True
        return self._engine

    def embed_image(self, image_path: Path) -> list[float]:
        image_path = image_path.expanduser().resolve()
        with self._lock:
            # LiteRT's native image decoder does not support WebP or HEIC.
            if image_path.suffix.lower() in {".webp", ".heic"}:
                if image_path.suffix.lower() == ".heic":
                    try:
                        from pillow_heif import register_heif_opener
                    except ImportError as exc:
                        raise RuntimeError(
                            "HEIC images require: python -m pip install -e '.[heic]'"
                        ) from exc
                    register_heif_opener()
                with NamedTemporaryFile(suffix=".png") as converted:
                    with Image.open(image_path) as image:
                        ImageOps.exif_transpose(image).convert("RGB").save(
                            converted.name, format="PNG"
                        )
                    content = self._content.ImageFile(converted.name)
                    return self._load().compute_embedding(content).embedding
            content = self._content.ImageFile(str(image_path))
            return self._load().compute_embedding(content).embedding

    def embed_text(self, text: str) -> list[float]:
        with self._lock:
            return self._load().compute_embedding(text).embedding

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            self.loaded = False
