import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from local_image_search.db import connect, needs_face_indexing
from local_image_search.face_detection import InsightFaceDetector
from local_image_search.index_service import index_roots
from local_image_search.stub_embedder import StubEmbedder


def test_insightface_model_identity_triggers_face_reindex(monkeypatch, tmp_path):
    analysis = MagicMock()
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=analysis))
    small = InsightFaceDetector(model_name="buffalo_s")
    large = InsightFaceDetector(model_name="buffalo_l")
    assert small.name == small.embedding_model == "insightface/buffalo_s"
    assert large.name == large.embedding_model == "insightface/buffalo_l"
    album = tmp_path / "album"
    album.mkdir()
    path = album / "portrait.jpg"
    Image.new("RGB", (24, 24), "white").save(path)
    monkeypatch.setattr(small, "detect_faces", MagicMock(return_value=[]))
    monkeypatch.setattr(large, "detect_faces", MagicMock(return_value=[]))
    db = tmp_path / "images.db"
    index_roots(db, [album], StubEmbedder(), small)
    with connect(db) as conn:
        assert not needs_face_indexing(conn, path, small.name, small.embedding_model)
        assert needs_face_indexing(conn, path, large.name, large.embedding_model)
    result = index_roots(db, [album], StubEmbedder(), large)
    assert result.indexed == 0
    large.detect_faces.assert_called_once_with(path)
