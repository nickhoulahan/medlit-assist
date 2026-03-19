import importlib
import sys
import types

import numpy as np
import pytest


def _load_embeddings_module_with_fake_model(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, values):
            if isinstance(values, str):
                return np.array([0.1, 0.2, 0.3], dtype=np.float32)
            return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    fake_sentence_transformers = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", fake_sentence_transformers
    )

    import src.medlit_agent.pmc_service.embeddings_service as embeddings_module

    return importlib.reload(embeddings_module)


def test_get_embeddings_returns_list_of_vectors(monkeypatch):
    embeddings_module = _load_embeddings_module_with_fake_model(monkeypatch)
    service = embeddings_module.SBertEmbeddingsService

    result = service.get_embeddings(["a", "b"])

    assert isinstance(result, list)
    assert np.allclose(np.asarray(result), np.asarray([[0.1, 0.2], [0.3, 0.4]]))


def test_get_embedding_returns_single_vector(monkeypatch):
    embeddings_module = _load_embeddings_module_with_fake_model(monkeypatch)
    service = embeddings_module.SBertEmbeddingsService

    result = service.get_embedding("query")

    assert isinstance(result, list)
    assert result == pytest.approx([0.1, 0.2, 0.3])
