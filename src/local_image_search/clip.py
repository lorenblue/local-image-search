from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path


class ClipEmbedder(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed_image(self, image_path: Path) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError


class StubClipEmbedder(ClipEmbedder):
    name = "stub-clip-v1"
    dimensions = 512

    def embed_image(self, image_path: Path) -> list[float]:
        return self._embed(image_path.stem.replace("-", " ").replace("_", " "))

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        for word in words:
            if not word:
                continue
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenClipEmbedder(ClipEmbedder):
    dimensions = 512

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP search requires: python -m pip install -e '.[ml]'"
            ) from exc

        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            pass
        else:
            register_heif_opener()

        self._image_class = Image
        self._open_clip = open_clip
        self._torch = torch
        self._device = os.environ.get("CLIP_DEVICE") or self._default_device(torch)
        model_name = os.environ.get("CLIP_MODEL", model_name)
        pretrained = os.environ.get("CLIP_PRETRAINED", pretrained)
        self.name = f"open-clip/{model_name}/{pretrained}"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def embed_image(self, image_path: Path) -> list[float]:
        image = self._image_class.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            features = self._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).cpu().tolist()]

    def embed_text(self, text: str) -> list[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with self._torch.inference_mode():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).cpu().tolist()]

    @staticmethod
    def _default_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def make_clip_embedder(name: str) -> ClipEmbedder:
    normalized = name.lower().strip()
    if normalized == "stub":
        return StubClipEmbedder()
    if normalized in {"open-clip", "openclip", "clip"}:
        return OpenClipEmbedder()
    raise ValueError(f"Unknown CLIP embedder: {name}")
