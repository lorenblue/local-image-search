from __future__ import annotations

import hashlib
import math
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenClipPreset:
    key: str
    title: str
    model_name: str
    pretrained: str
    dimensions: int


DEFAULT_OPEN_CLIP_PRESET = "fast"
OPEN_CLIP_PRESET_ALIASES = {
    "best": "better",
}
OPEN_CLIP_PRESETS = {
    "fast": OpenClipPreset(
        key="fast",
        title="Fast",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        dimensions=512,
    ),
    "better": OpenClipPreset(
        key="better",
        title="Better",
        model_name="ViT-B-16",
        pretrained="datacomp_xl_s13b_b90k",
        dimensions=512,
    ),
}


@dataclass(frozen=True)
class OpenClipModelConfig:
    model_name: str
    pretrained: str
    model_preset: str
    dimensions: int

    @property
    def name(self) -> str:
        return f"open-clip/{self.model_name}/{self.pretrained}"


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
    model_preset = "stub"

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
    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        model_preset: str | None = None,
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
        config = resolve_open_clip_model_config(
            model_preset=model_preset,
            model_name=model_name,
            pretrained=pretrained,
        )
        self.name = config.name
        self.model_preset = config.model_preset
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
            device=self._device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(config.model_name)
        self.dimensions = self._detect_dimensions()

    def embed_image(self, image_path: Path) -> list[float]:
        image = self._image_class.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            features = self._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).cpu().tolist()]

    def _detect_dimensions(self) -> int:
        embed_dim = getattr(self._model, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
        with self._torch.inference_mode():
            features = self._model.encode_text(
                self._tokenizer(["dimension check"]).to(self._device)
            )
        return int(features.shape[-1])

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


class LazyClipEmbedder(ClipEmbedder):
    def __init__(
        self,
        config: OpenClipModelConfig,
    ) -> None:
        self.name = config.name
        self.dimensions = config.dimensions
        self.model_preset = config.model_preset
        self.loaded = False
        self._config = config
        self._lock = threading.Lock()
        self._embedder: OpenClipEmbedder | None = None

    def embed_image(self, image_path: Path) -> list[float]:
        return self._load().embed_image(image_path)

    def embed_text(self, text: str) -> list[float]:
        return self._load().embed_text(text)

    def _load(self) -> OpenClipEmbedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = OpenClipEmbedder(
                    model_name=self._config.model_name,
                    pretrained=self._config.pretrained,
                    model_preset=self._config.model_preset,
                )
                self.loaded = True
                self.dimensions = self._embedder.dimensions
            return self._embedder


def resolve_open_clip_model_config(
    *,
    model_preset: str | None = None,
    model_name: str | None = None,
    pretrained: str | None = None,
) -> OpenClipModelConfig:
    model_preset = os.environ.get("CLIP_MODEL_PRESET", model_preset or "")
    model_name = os.environ.get("CLIP_MODEL", model_name or "")
    pretrained = os.environ.get("CLIP_PRETRAINED", pretrained or "")

    normalized_preset = (model_preset or DEFAULT_OPEN_CLIP_PRESET).lower().strip()
    normalized_preset = OPEN_CLIP_PRESET_ALIASES.get(
        normalized_preset,
        normalized_preset,
    )
    if normalized_preset not in OPEN_CLIP_PRESETS:
        raise ValueError(f"Unknown OpenCLIP model preset: {model_preset}")

    preset = OPEN_CLIP_PRESETS[normalized_preset]
    return OpenClipModelConfig(
        model_name=model_name or preset.model_name,
        pretrained=pretrained or preset.pretrained,
        model_preset=normalized_preset,
        dimensions=preset.dimensions,
    )


def make_clip_embedder(
    name: str,
    *,
    model_preset: str | None = None,
    model_name: str | None = None,
    pretrained: str | None = None,
    lazy: bool = False,
) -> ClipEmbedder:
    normalized = name.lower().strip()
    if normalized == "stub":
        return StubClipEmbedder()
    if normalized in {"open-clip", "openclip", "clip"}:
        if lazy:
            return LazyClipEmbedder(
                resolve_open_clip_model_config(
                    model_preset=model_preset,
                    model_name=model_name,
                    pretrained=pretrained,
                )
            )
        return OpenClipEmbedder(
            model_preset=model_preset,
            model_name=model_name,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown CLIP embedder: {name}")
