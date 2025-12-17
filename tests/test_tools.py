from unittest.mock import patch

import pytest

from src.tools import search_pubmed_central, tools


class TestSearchPubmedCentral:

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_success(self, mock_fetch):
        # Mock the PMC endpoint response
        mock_fetch.return_value = [
            {
                "pmcid": "12345",
                "apa_citation": "Smith, J. (2024). Test Article. Journal, 10(1), 1-10. https://doi.org/10.1234/test",
                "abstract": "This is a test abstract.",
            },
            {
                "pmcid": "67890",
                "apa_citation": "Doe, A. (2023). Another Article. Science, 5(2), 20-30. https://doi.org/10.5678/test",
                "abstract": "Another test abstract.",
            },
        ]

        result = search_pubmed_central.invoke({"query": "test query", "max_results": 2})

        assert len(result) == 2
        assert result[0]["pmcid"] == "12345"
        assert result[0]["citation"] == "Smith, J. (2024). Test Article. Journal, 10(1), 1-10. https://doi.org/10.1234/test"
        assert result[0]["abstract"] == "This is a test abstract."
        assert result[1]["pmcid"] == "67890"
        mock_fetch.assert_called_once_with("test query", retmax=2)

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_default_max_results(self, mock_fetch):
        mock_fetch.return_value = []

        search_pubmed_central.invoke({"query": "test"})

        mock_fetch.assert_called_once_with("test", retmax=3)

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_empty_results(self, mock_fetch):
        mock_fetch.return_value = []

        result = search_pubmed_central.invoke({"query": "nonexistent query"})

        assert result == []
        assert isinstance(result, list)

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_custom_max_results(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "pmcid": str(i),
                "apa_citation": f"Citation {i}",
                "abstract": f"Abstract {i}",
            }
            for i in range(5)
        ]

        result = search_pubmed_central.invoke({"query": "test", "max_results": 5})

        assert len(result) == 5
        mock_fetch.assert_called_once_with("test", retmax=5)

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_error_handling(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Error searching PubMed Central: Network error"):
            search_pubmed_central.invoke({"query": "test"})

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_malformed_response(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "pmcid": "12345",
                "apa_citation": "Citation",
                "abstract": "Abstract",
            }
        ]

        result = search_pubmed_central.invoke({"query": "test"})

        assert len(result) == 1
        assert "pmcid" in result[0]
        assert "citation" in result[0]
        assert "abstract" in result[0]

    def test_search_pubmed_central_tool_metadata(self):
        assert search_pubmed_central.name == "search_pubmed_central"
        assert "PubMed Central" in search_pubmed_central.description
        assert hasattr(search_pubmed_central, "invoke")

    @patch("src.tools.PMCEndpoint.fetch_pmc_records")
    def test_search_pubmed_central_result_structure(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "pmcid": "12345",
                "apa_citation": "Test Citation",
                "abstract": "Test Abstract",
            }
        ]

        result = search_pubmed_central.invoke({"query": "test"})

        assert isinstance(result, list)
        assert len(result) == 1
        assert set(result[0].keys()) == {"pmcid", "citation", "abstract"}


class TestToolsExport:
    def test_tools_list_contains_search_pubmed_central(self):
        assert search_pubmed_central in tools

    def test_tools_list_length(self):
        assert len(tools) == 1

    def test_all_tools_are_callable(self):
        for tool in tools:
            assert hasattr(tool, "invoke")
            assert hasattr(tool, "name")
