from unittest.mock import MagicMock, patch

from src.medlit_agent.pmc_service.full_text_retriever import FullTextRetriever


@patch("src.medlit_agent.pmc_service.full_text_retriever.ChromaDB")
def test_retrieve_full_text_uses_chroma_cache_first(mock_chroma_db):
    mock_db = MagicMock()
    mock_db.document_exists.return_value = True
    mock_db.get_sections_by_pmcid.return_value = [
        {"title": "Methods", "body": "cached chunk"}
    ]
    mock_chroma_db.return_value = mock_db

    retriever = FullTextRetriever()

    with patch(
        "src.medlit_agent.pmc_service.full_text_retriever.PMCEndpoint.fetch_pmcid_xml"
    ) as mock_fetch_xml:
        result = retriever.retrieve_full_text("PMC123")

    assert result == [{"title": "Methods", "body": "cached chunk"}]
    mock_db.document_exists.assert_called_once_with("PMC123")
    mock_db.get_sections_by_pmcid.assert_called_once_with("PMC123", limit=5)
    mock_fetch_xml.assert_not_called()


@patch("src.medlit_agent.pmc_service.full_text_retriever.ChromaDB")
@patch("src.medlit_agent.pmc_service.full_text_retriever.PMCEndpoint.fetch_pmcid_xml")
def test_retrieve_full_text_fetches_stores_then_reads_cache(
    mock_fetch_xml, mock_chroma_db
):
    mock_db = MagicMock()
    mock_db.document_exists.return_value = False
    mock_db.get_sections_by_pmcid.return_value = [
        {"title": "Results", "body": "chunk 1"},
        {"title": "Discussion", "body": "chunk 2"},
    ]
    mock_chroma_db.return_value = mock_db
    mock_fetch_xml.return_value = "<xml/>"

    retriever = FullTextRetriever()
    retriever.converter.convert = MagicMock(
        return_value=[
            {"title": "Results", "body": "full section content"},
            {"title": "Discussion", "body": "more content"},
        ]
    )

    result = retriever.retrieve_full_text("PMC999", n_results=5)

    assert result == [
        {"title": "Results", "body": "chunk 1"},
        {"title": "Discussion", "body": "chunk 2"},
    ]
    mock_db.document_exists.assert_called_once_with("PMC999")
    mock_fetch_xml.assert_called_once_with("PMC999")
    mock_db.add.assert_called_once_with(
        "PMC999",
        [
            {"title": "Results", "body": "full section content"},
            {"title": "Discussion", "body": "more content"},
        ],
    )
    mock_db.get_sections_by_pmcid.assert_called_once_with("PMC999", limit=5)


@patch("src.medlit_agent.pmc_service.full_text_retriever.ChromaDB")
def test_retrieve_full_text_respects_requested_top_n(mock_chroma_db):
    mock_db = MagicMock()
    mock_db.document_exists.return_value = True
    mock_db.get_sections_by_pmcid.return_value = []
    mock_chroma_db.return_value = mock_db

    retriever = FullTextRetriever()
    retriever.retrieve_full_text("PMC555", n_results=3)

    mock_db.get_sections_by_pmcid.assert_called_once_with("PMC555", limit=3)
