from src.medlit_agent.pmc_service.embeddings_service import SBertEmbeddingsService
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint
from src.medlit_agent.pmc_service.xml_to_dict import XMLToDictConverter


def test_live_list_embeddings_service():
    pmcid = "PMC9759163"
    xml_content = PMCEndpoint.fetch_pmcid_xml(pmcid)
    sections = XMLToDictConverter.convert(xml_content)
    section_count = len(sections)
    embedding_service = SBertEmbeddingsService()
    embeddings = embedding_service.get_embeddings(
        [section.get("text", "") for section in sections]
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == section_count
    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert (
            len(embedding) == 384
        )  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert all(isinstance(value, float) for value in embedding)


def test_live_single_embedding_service():
    query = "diabetes and obesity"
    embedding_service = SBertEmbeddingsService()
    embedding = embedding_service.get_embedding(query)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert len(embedding) == 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    assert all(isinstance(value, float) for value in embedding)
