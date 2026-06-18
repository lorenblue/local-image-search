from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class Captioner(ABC):
    name: str

    @abstractmethod
    def caption(self, image_path: Path) -> str:
        raise NotImplementedError


class StubCaptioner(Captioner):
    name = "stub-captioner-v1"

    def caption(self, image_path: Path) -> str:
        tokens = (
            image_path.stem.replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
            .lower()
            .split()
        )
        readable_name = " ".join(tokens) or image_path.name.lower()
        return (
            f"A local image named {readable_name}. "
            "This stub caption is generated from the filename for offline smoke testing."
        )


class MoondreamCaptioner(Captioner):
    default_model = "moondream2"
    search_caption_prompt = (
        "Describe this image for semantic photo search. Include people, clothing, "
        "visible text, signs, logos, objects, vehicles, actions, location, colors, "
        "and scene context. Be literal and specific."
    )

    def __init__(self) -> None:
        try:
            import moondream as md
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Moondream captioning requires: python -m pip install -e '.[ml]'"
            ) from exc

        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            pass
        else:
            register_heif_opener()

        self._image_class = Image
        model_name = os.environ.get("MOONDREAM_MODEL", self.default_model)
        self.name = f"moondream-local/{model_name}"
        kwargs = {
            "local": True,
            "model": model_name,
            "max_batch_size": 1,
        }
        api_key = os.environ.get("MOONDREAM_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        try:
            self._model = md.vl(**kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Moondream local setup failed. If this is the first model download, "
                "set MOONDREAM_API_KEY and rerun; cached local inference should work "
                "offline after setup."
            ) from exc

    def caption(self, image_path: Path) -> str:
        image = self._image_class.open(image_path)
        result = self._model.query(image=image, question=self.search_caption_prompt)
        caption = result.get("answer") if isinstance(result, dict) else str(result)
        if not caption:
            result = self._model.caption(image, length="long")
            caption = result.get("caption") if isinstance(result, dict) else str(result)
        if not caption:
            raise RuntimeError(f"Moondream returned an empty caption for {image_path}")
        return str(caption)


def make_captioner(name: str) -> Captioner:
    normalized = name.lower().strip()
    if normalized == "stub":
        return StubCaptioner()
    if normalized == "moondream":
        return MoondreamCaptioner()
    raise ValueError(f"Unknown captioner: {name}")
