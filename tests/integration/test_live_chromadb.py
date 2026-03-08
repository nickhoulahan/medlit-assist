from pathlib import Path

from src.medlit_agent.pmc_service.chroma_db import ChromaDB
from src.medlit_agent.pmc_service.embeddings_service import SBertEmbeddingsService
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint
from src.medlit_agent.pmc_service.xml_to_dict import XMLToDictConverter


def _setup_test_collection_from_xml(
    collection_name: str,
    persist_directory: Path,
    pmcid: str,
) -> ChromaDB:
    service = ChromaDB(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    xml_content = PMCEndpoint.fetch_pmcid_xml(pmcid)
    sections = XMLToDictConverter.convert(xml_content)
    service.add(pmcid=pmcid, texts=sections)
    return service


def _teardown_test_collection(service: ChromaDB, collection_name: str) -> None:
    service.client.delete_collection(name=collection_name)


def test_live_chromadb_from_xml(tmp_path: Path):
    pmcid = "PMC9759163"
    collection_name = f"test_collection_{pmcid.lower()}"

    service = _setup_test_collection_from_xml(
        collection_name=collection_name,
        persist_directory=tmp_path,
        pmcid=pmcid,
    )

    try:
        query_embedding = SBertEmbeddingsService.get_embedding("diabetes")
        results = service.query(query_embedding=query_embedding, n_results=3)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert isinstance(result, dict)
            assert result.get("pmcid") == pmcid
            assert "title" in result
            assert "text" in result
            # assert that "diabtetes" is in either the title or text of the result (case-insensitive)
            title = result.get("title", "").lower()
            text = result.get("text", "").lower()
            assert "diabetes" in title or "diabetes" in text
    finally:
        _teardown_test_collection(service, collection_name)
