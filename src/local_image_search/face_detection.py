from __future__ import annotations

import contextlib
import io
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageOps

from local_image_search.models import FaceBox

DEFAULT_FACE_DETECTOR = "insightface"


class FaceDetector(ABC):
    name: str

    @property
    def embedding_model(self) -> str:
        return self.name

    @abstractmethod
    def detect_faces(self, image_path: Path) -> list[FaceBox]:
        raise NotImplementedError


class InsightFaceDetector(FaceDetector):
    name = "insightface"

    def __init__(
        self,
        *,
        model_name: str = "buffalo_s",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            import numpy as np
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace detection requires: python -m pip install -e '.[face]'"
            ) from exc

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                app = FaceAnalysis(
                    name=model_name,
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=-1, det_size=det_size)
        except Exception as exc:
            raise RuntimeError(
                "InsightFace could not load its local model. "
                "Run this once while online so the model can download, then retry: "
                "image-search index ~/Pictures"
            ) from exc

        self._app = app
        self._np = np

    def detect_faces(self, image_path: Path) -> list[FaceBox]:
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            pass
        else:
            register_heif_opener()

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_array = self._np.asarray(image)[:, :, ::-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            faces = self._app.get(image_array)
        boxes: list[FaceBox] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            embedding = getattr(face, "normed_embedding", None)
            boxes.append(
                FaceBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    detection_score=float(face.det_score),
                    embedding=(
                        None
                        if embedding is None
                        else [float(value) for value in embedding.tolist()]
                    ),
                )
            )
        return boxes


def make_face_detector(name: str) -> FaceDetector:
    normalized = name.lower().strip()
    if normalized == "insightface":
        return InsightFaceDetector()
    raise ValueError(f"Unknown face detector: {name}")
