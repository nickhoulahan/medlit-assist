from pathlib import Path

import pytest

from src.medlit_agent.pmc_service.chroma_db import ChromaDB
from src.medlit_agent.pmc_service.full_text_retriever import FullTextRetriever


def test_live_full_text_retriever_pmc1831666(tmp_path: Path):
    pmcid = "PMC1831666"
    collection_name = f"test_full_text_retriever_{pmcid.lower()}"

    retriever = FullTextRetriever()
    retriever.db = ChromaDB(
        collection_name=collection_name,
        persist_directory=tmp_path,
    )

    try:
        with pytest.raises(ValueError, match="No <body> element found"):
            retriever.retrieve_full_text(pmcid, n_results=5)
    finally:
        retriever.db.client.delete_collection(name=collection_name)


def test_live_full_text_retriever_success_pmc9759163(tmp_path: Path):
    pmcid = "PMC9759163"
    collection_name = f"test_full_text_retriever_{pmcid.lower()}"

    retriever = FullTextRetriever()
    retriever.db = ChromaDB(
        collection_name=collection_name,
        persist_directory=tmp_path,
    )

    try:
        sections = retriever.retrieve_full_text(pmcid, n_results=5)

        assert isinstance(sections, list)
        assert len(sections) > 0

        for section in sections:
            assert isinstance(section, dict)
            assert "title" in section
            assert isinstance(section["title"], str)
            assert section["title"].strip() != ""

            assert "body" in section
            assert isinstance(section["body"], str)
            assert section["body"].strip() != ""
    finally:
        retriever.db.client.delete_collection(name=collection_name)
