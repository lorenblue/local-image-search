import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from local_image_search import cli, gemma_embedder
from local_image_search.gemma_embedder import EmbeddingGemma2Embedder
from local_image_search.server import create_app


@pytest.fixture
def litert(monkeypatch, tmp_path):
    module = SimpleNamespace(
        Backend=SimpleNamespace(GPU=MagicMock()),
        Content=SimpleNamespace(ImageFile=MagicMock()),
        EmbeddingEngine=MagicMock(),
    )
    module.EmbeddingEngine.return_value.compute_embedding.return_value.embedding = [0.5] * 768
    monkeypatch.setitem(sys.modules, "litert_lm", module)
    model = tmp_path / "embeddinggemma-2-text-vision-440m.litertlm"
    model.touch()
    return module, model


def test_gemma_routes_image_and_text_reuses_engine_and_closes(litert, tmp_path):
    module, model = litert
    engine = module.EmbeddingEngine.return_value
    with EmbeddingGemma2Embedder(model) as embedder:
        assert embedder.dimensions == 768
        assert embedder.name == "embeddinggemma-2/text-vision-440m"
        image = tmp_path / "image.jpg"
        assert embedder.embed_image(image) == [0.5] * 768
        module.Content.ImageFile.assert_called_once_with(str(image))
        engine.compute_embedding.assert_called_with(module.Content.ImageFile.return_value)
        assert embedder.embed_text("red apple") == [0.5] * 768
        engine.compute_embedding.assert_called_with("red apple")
        module.EmbeddingEngine.assert_called_once_with(
            str(model), backend=module.Backend.GPU.return_value,
            vision_backend=module.Backend.GPU.return_value,
        )
    embedder.close()
    engine.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        embedder.embed_text("after close")


def test_gemma_lazy_loading_and_server_disposal(litert, tmp_path):
    module, model = litert
    embedder = EmbeddingGemma2Embedder(model)
    assert not embedder.loaded
    module.EmbeddingEngine.assert_not_called()
    app = create_app(tmp_path / "images.db", embedder)
    assert app.state.search_service.embedder is embedder
    assert app.state.index_service.embedder is embedder
    with TestClient(app):
        embedder.embed_text("query")
        assert embedder.loaded
    module.EmbeddingEngine.return_value.close.assert_called_once()


@pytest.mark.parametrize("fails", [False, True])
def test_gemma_converts_webp_and_removes_temporary_file(litert, tmp_path, fails):
    module, model = litert
    image_path = tmp_path / "image.webp"
    Image.new("RGB", (24, 24), "red").save(image_path)
    converted_paths = []

    def image_file(path):
        converted_paths.append(Path(path))
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.size == (24, 24)
        return path

    module.Content.ImageFile.side_effect = image_file
    if fails:
        module.EmbeddingEngine.return_value.compute_embedding.side_effect = RuntimeError("test")
    with EmbeddingGemma2Embedder(model) as embedder:
        if fails:
            with pytest.raises(RuntimeError, match="test"):
                embedder.embed_image(image_path)
        else:
            assert embedder.embed_image(image_path) == [0.5] * 768
    assert len(converted_paths) == 1
    assert not converted_paths[0].exists()
    assert image_path.exists()


@pytest.mark.parametrize("command", ["index", "search", "similar", "serve"])
def test_cli_gemma_configuration(command, litert):
    _, model = litert
    positional = [] if command == "serve" else ["example"]
    args = cli.build_parser().parse_args(
        [command, *positional, "--gemma-model", str(model)]
    )
    with EmbeddingGemma2Embedder(args.gemma_model) as embedder:
        assert isinstance(embedder, EmbeddingGemma2Embedder)



def test_gemma_default_model_is_independent_of_working_directory(monkeypatch, litert, tmp_path):
    module, model = litert
    monkeypatch.delenv("GEMMA_MODEL_PATH", raising=False)
    monkeypatch.setattr(gemma_embedder, "DEFAULT_GEMMA_MODEL_PATH", model)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    args = cli.build_parser().parse_args(["search", "apple"])
    with EmbeddingGemma2Embedder(args.gemma_model) as embedder:
        assert embedder.dimensions == 768
        embedder.embed_text("apple")
    assert module.EmbeddingEngine.call_args.args == (str(model),)


def test_gemma_model_environment_and_missing_model(monkeypatch, litert, tmp_path):
    _, model = litert
    monkeypatch.delenv("GEMMA_MODEL_PATH", raising=False)
    monkeypatch.setattr(gemma_embedder, "DEFAULT_GEMMA_MODEL_PATH", tmp_path / "missing.litertlm")
    with pytest.raises(FileNotFoundError, match="Place embeddinggemma"):
        EmbeddingGemma2Embedder()
    monkeypatch.setenv("GEMMA_MODEL_PATH", str(model))
    with EmbeddingGemma2Embedder() as embedder:
        assert embedder.dimensions == 768


def test_cli_disposes_gemma_on_failure(monkeypatch, litert):
    module, model = litert
    service = MagicMock()

    def failing_search(query, limit):
        service.call_args.args[1].embed_text(query)
        raise RuntimeError("test failure")

    service.return_value.search.side_effect = failing_search
    monkeypatch.setattr(cli, "SearchService", service)
    assert cli.main(
        ["search", "apple", "--gemma-model", str(model)]
    ) == 1
    module.EmbeddingEngine.return_value.close.assert_called_once()
