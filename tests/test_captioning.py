from __future__ import annotations

from pathlib import Path

from local_image_search.captioning import MoondreamCaptioner


class FakeImage:
    pass


class FakeImageClass:
    opened_path: Path | None = None

    @classmethod
    def open(cls, image_path: Path) -> FakeImage:
        cls.opened_path = image_path
        return FakeImage()


class FakeMoondreamModel:
    question: str | None = None

    def query(self, *, image: FakeImage, question: str) -> dict[str, str]:
        self.question = question
        return {"answer": "A person wearing glasses beside a red car."}


def test_moondream_captioner_uses_search_oriented_query_prompt() -> None:
    captioner = MoondreamCaptioner.__new__(MoondreamCaptioner)
    captioner._image_class = FakeImageClass
    captioner._model = FakeMoondreamModel()

    caption = captioner.caption(Path("photo.jpg"))

    assert caption == "A person wearing glasses beside a red car."
    assert FakeImageClass.opened_path == Path("photo.jpg")
    assert captioner._model.question == MoondreamCaptioner.search_caption_prompt
