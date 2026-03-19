import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.medlit_agent.pmc_service.chroma_db import ChromaDB


def test_init_uses_persist_directory(tmp_path):
    created = {}

    class FakeClient:
        def __init__(self, path):
            created["path"] = path

        def get_or_create_collection(self, name):
            created["collection_name"] = name
            return MagicMock()

    fake_module = SimpleNamespace(PersistentClient=FakeClient)

    with patch.dict(sys.modules, {"chromadb": fake_module}):
        ChromaDB(collection_name="unit_test_collection", persist_directory=tmp_path)

    assert created["path"] == str(tmp_path)
    assert created["collection_name"] == "unit_test_collection"


def test_split_text_handles_empty_and_overlap():
    assert ChromaDB._split_text("") == []

    chunks = ChromaDB._split_text("abcdefghij", chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_add_chunks_and_stores_embeddings():
    db = ChromaDB.__new__(ChromaDB)
    db.collection = MagicMock()

    with patch(
        "src.medlit_agent.pmc_service.chroma_db.SBertEmbeddingsService"
    ) as mock_embed_cls:
        mock_embedder = mock_embed_cls.return_value
        mock_embedder.get_embeddings.return_value = [[0.1], [0.2]]

        with patch.object(ChromaDB, "_split_text", return_value=["chunk a", "chunk b"]):
            db.add("PMC42", [{"title": "Results", "body": "Long section text"}])

    mock_embedder.get_embeddings.assert_called_once_with(["chunk a", "chunk b"])
    kwargs = db.collection.add.call_args.kwargs
    assert kwargs["ids"] == ["PMC42_0", "PMC42_1"]
    assert kwargs["embeddings"] == [[0.1], [0.2]]
    assert kwargs["metadatas"] == [
        {"title": "Results", "text": "chunk a", "pmcid": "PMC42"},
        {"title": "Results", "text": "chunk b", "pmcid": "PMC42"},
    ]


def test_query_document_exists_and_get_sections():
    db = ChromaDB.__new__(ChromaDB)
    db.collection = MagicMock()

    db.collection.query.return_value = {
        "metadatas": [[{"title": "Intro", "text": "alpha", "pmcid": "PMC1"}]]
    }
    result = db.query([0.5], n_results=2, pmcid="PMC1")
    assert result == [{"title": "Intro", "text": "alpha", "pmcid": "PMC1"}]

    query_kwargs = db.collection.query.call_args.kwargs
    assert query_kwargs["where"] == {"pmcid": "PMC1"}

    db.collection.query.return_value = {"metadatas": []}
    assert db.query([0.2], n_results=1) == []

    db.collection.get.return_value = {"ids": ["PMC1_0"]}
    assert db.document_exists("PMC1") is True

    db.collection.get.return_value = {
        "metadatas": [
            {"title": "Methods", "text": "method text"},
            {"text": "untitled body"},
        ]
    }
    sections = db.get_sections_by_pmcid("PMC1", limit=2)
    assert sections == [
        {"title": "Methods", "body": "method text"},
        {"title": "Untitled Section", "body": "untitled body"},
    ]
