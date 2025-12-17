import os
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from src.pmc_endpoint import PMCEndpoint


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv("EMAIL", "test@example.com")
    monkeypatch.setenv("PMC_API_KEY", "test_api_key")
    # Re-import to apply env vars
    import importlib
    from src import pmc_endpoint
    importlib.reload(pmc_endpoint)


@pytest.fixture
def sample_esearch_response():
    """Mock XML response from Entrez.esearch"""
    return """<?xml version="1.0"?>
    <!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD eSearchResult, 11 May 2002//EN" "https://www.ncbi.nlm.nih.gov/entrez/query/DTD/eSearchResult.dtd">
    <eSearchResult>
        <IdList>
            <Id>12345678</Id>
            <Id>87654321</Id>
        </IdList>
    </eSearchResult>"""


@pytest.fixture
def sample_article_xml():
    """Sample PMC article XML for testing parse_article"""
    return """<?xml version="1.0"?>
    <!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.0 20120330//EN" "JATS-archivearticle1.dtd">
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="pmc">12345678</article-id>
                <article-id pub-id-type="doi">10.1234/example.2024.001</article-id>
                <title-group>
                    <article-title>Hyperspectral Imaging for Blood Oxygen Monitoring</article-title>
                </title-group>
                <contrib-group>
                    <contrib contrib-type="author">
                        <name>
                            <surname>Smith</surname>
                            <given-names>J. A.</given-names>
                        </name>
                    </contrib>
                    <contrib contrib-type="author">
                        <name>
                            <surname>Johnson</surname>
                            <given-names>B. C.</given-names>
                        </name>
                    </contrib>
                </contrib-group>
                <pub-date pub-type="epub">
                    <year>2024</year>
                </pub-date>
                <journal-meta>
                    <journal-title-group>
                        <journal-title>Journal of Medical Imaging</journal-title>
                    </journal-title-group>
                </journal-meta>
                <volume>15</volume>
                <issue>3</issue>
                <fpage>100</fpage>
                <lpage>115</lpage>
                <abstract>
                    <p>This study investigates hyperspectral imaging techniques.</p>
                    <p>We found significant improvements in blood oxygen monitoring.</p>
                </abstract>
            </article-meta>
        </front>
    </article>"""


class TestPMCEndpointInit:

    def test_initialization_with_env_vars(self, mock_env_vars):
        assert PMCEndpoint.endpoint.email == "test@example.com"
        assert PMCEndpoint.endpoint.tool == "pmc_apa_abstract_fetcher"
        assert PMCEndpoint.endpoint.api_key == "test_api_key"

    def test_initialization_without_email(self, monkeypatch):
        monkeypatch.delenv("EMAIL", raising=False)
        monkeypatch.delenv("PMC_API_KEY", raising=False)
        # Re-import to trigger class-level initialization
        import importlib

        from src import pmc_endpoint

        importlib.reload(pmc_endpoint)
        # When EMAIL env var is missing, it should be None or empty
        email = pmc_endpoint.PMCEndpoint.endpoint.email
        assert email is None or email == "" or email == os.getenv("EMAIL")


class TestFetchPMCIds:

    @patch("src.pmc_endpoint.Entrez.esearch")
    @patch("src.pmc_endpoint.Entrez.read")
    def test_fetch_pmc_ids_success(
        self, mock_read, mock_esearch, mock_env_vars, sample_esearch_response
    ):
        # Mock the Entrez API calls
        mock_esearch.return_value = MagicMock()
        mock_read.return_value = {"IdList": ["12345678", "87654321"]}

        query = "hyperspectral imaging"
        ids = PMCEndpoint._fetch_pmc_ids(query, retmax=2)

        assert ids == ["12345678", "87654321"]
        mock_esearch.assert_called_once()
        mock_read.assert_called_once()

    @patch("src.pmc_endpoint.Entrez.esearch")
    @patch("src.pmc_endpoint.Entrez.read")
    def test_fetch_pmc_ids_empty_result(self, mock_read, mock_esearch, mock_env_vars):

        mock_esearch.return_value = MagicMock()
        mock_read.return_value = {"IdList": []}

        ids = PMCEndpoint._fetch_pmc_ids("nonexistent_query")

        assert ids == []
        mock_esearch.assert_called_once()

    @patch("src.pmc_endpoint.Entrez.esearch")
    def test_fetch_pmc_ids_network_error(self, mock_esearch, mock_env_vars):

        mock_esearch.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            PMCEndpoint._fetch_pmc_ids("test query")

    @patch("src.pmc_endpoint.Entrez.esearch")
    @patch("src.pmc_endpoint.Entrez.read")
    def test_fetch_pmc_ids_custom_retmax(self, mock_read, mock_esearch, mock_env_vars):

        mock_esearch.return_value = MagicMock()
        mock_read.return_value = {"IdList": ["1", "2", "3", "4", "5"]}

        ids = PMCEndpoint._fetch_pmc_ids("test", retmax=5)

        assert len(ids) == 5
        # Test retmax was passed to esearch
        call_kwargs = mock_esearch.call_args[1]
        assert call_kwargs["retmax"] == 5


class TestParseArticle:


    def test_parse_article_complete(self, sample_article_xml):
        root = ET.fromstring(sample_article_xml)
        result = PMCEndpoint._parse_article(root, "12345678")

        assert result["pmcid"] == "12345678"
        assert "Smith, J." in result["apa_citation"]
        assert "Johnson, B." in result["apa_citation"]
        assert "(2024)" in result["apa_citation"]
        assert "Hyperspectral Imaging for Blood Oxygen Monitoring" in result[
            "apa_citation"
        ]
        assert "Journal of Medical Imaging" in result["apa_citation"]
        assert "15(3)" in result["apa_citation"]
        assert "100–115" in result["apa_citation"]  # En-dash
        assert "https://doi.org/10.1234/example.2024.001" in result["apa_citation"]
        assert "hyperspectral imaging" in result["abstract"].lower()

    def test_parse_article_single_author(self):
        xml = """<?xml version="1.0"?>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">999</article-id>
                    <title-group>
                        <article-title>Test Article</article-title>
                    </title-group>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Doe</surname>
                                <given-names>J.</given-names>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date pub-type="epub">
                        <year>2024</year>
                    </pub-date>
                    <journal-meta>
                        <journal-title-group>
                            <journal-title>Test Journal</journal-title>
                        </journal-title-group>
                    </journal-meta>
                    <volume>1</volume>
                    <abstract>
                        <p>Test abstract.</p>
                    </abstract>
                </article-meta>
            </front>
        </article>"""

        root = ET.fromstring(xml)
        result = PMCEndpoint._parse_article(root, "999")

        assert result["apa_citation"].startswith("Doe, J. (2024)")
        assert "&" not in result["apa_citation"]  # No ampersand for single author

    def test_parse_article_multiple_authors(self):
        xml = """<?xml version="1.0"?>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">888</article-id>
                    <title-group>
                        <article-title>Multi-Author Study</article-title>
                    </title-group>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Adams</surname>
                                <given-names>A.</given-names>
                            </name>
                        </contrib>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Baker</surname>
                                <given-names>B.</given-names>
                            </name>
                        </contrib>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Carter</surname>
                                <given-names>C.</given-names>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date pub-type="epub">
                        <year>2023</year>
                    </pub-date>
                    <journal-meta>
                        <journal-title-group>
                            <journal-title>Test Journal</journal-title>
                        </journal-title-group>
                    </journal-meta>
                    <volume>10</volume>
                    <abstract><p>Test.</p></abstract>
                </article-meta>
            </front>
        </article>"""

        root = ET.fromstring(xml)
        result = PMCEndpoint._parse_article(root, "888")

        assert "Adams, A., Baker, B., & Carter, C." in result["apa_citation"]

    def test_parse_article_missing_optional_fields(self):
        xml = """<?xml version="1.0"?>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">777</article-id>
                    <title-group>
                        <article-title>Minimal Article</article-title>
                    </title-group>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Test</surname>
                                <given-names>T.</given-names>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date pub-type="epub">
                        <year>2024</year>
                    </pub-date>
                    <journal-meta>
                        <journal-title-group>
                            <journal-title>Test Journal</journal-title>
                        </journal-title-group>
                    </journal-meta>
                    <volume>5</volume>
                    <abstract><p>Test.</p></abstract>
                </article-meta>
            </front>
        </article>"""

        root = ET.fromstring(xml)
        result = PMCEndpoint._parse_article(root, "777")

        # Should not raise error, citation should still be valid
        assert "Test, T. (2024)" in result["apa_citation"]
        assert "5" in result["apa_citation"]  # Volume without issue

    def test_parse_article_no_abstract(self):
        xml = """<?xml version="1.0"?>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">666</article-id>
                    <title-group>
                        <article-title>No Abstract Article</article-title>
                    </title-group>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Author</surname>
                                <given-names>A.</given-names>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date pub-type="epub">
                        <year>2024</year>
                    </pub-date>
                    <journal-meta>
                        <journal-title-group>
                            <journal-title>Test Journal</journal-title>
                        </journal-title-group>
                    </journal-meta>
                    <volume>1</volume>
                </article-meta>
            </front>
        </article>"""

        root = ET.fromstring(xml)
        result = PMCEndpoint._parse_article(root, "666")

        assert result["abstract"] == ""


class TestCleanAbstract:

    def test_clean_abstract_whitespace_normalization(self):
        raw = "This  is    a  test.\n\n\n\nAnother paragraph."
        cleaned = PMCEndpoint._clean_abstract(raw)

        assert "  " not in cleaned  # No double spaces
        assert "\n\n\n" not in cleaned  # Max two newlines

    def test_clean_abstract_chemical_notation(self):
        # chemical notation can cause issues if not formatted properly
        raw = "The study measured SO 2 levels in the atmosphere."
        cleaned = PMCEndpoint._clean_abstract(raw)

        assert "SO₂" in cleaned
        assert "SO 2" not in cleaned

    def test_clean_abstract_structured_headers(self):
        raw = "Objective: To test something. Methods: We did tests. Results: Found results."
        cleaned = PMCEndpoint._clean_abstract(raw)

        # First header won't have \n\n prefix, subsequent ones will
        assert "Objective:" in cleaned
        assert "\n\nMethods:" in cleaned
        assert "\n\nResults:" in cleaned

    def test_clean_abstract_all_headers(self):
        headers = [
            "Objective",
            "Impact Statement",
            "Introduction",
            "Methods",
            "Results",
            "Conclusion",
        ]
        raw = " ".join([f"{h}: Test." for h in headers])
        cleaned = PMCEndpoint._clean_abstract(raw)

        # Verify all headers are present (may or may not have \n\n prefix depending on position)
        for h in headers:
            assert f"{h}:" in cleaned

    def test_clean_abstract_empty_string(self):
        assert PMCEndpoint._clean_abstract("") == ""

    def test_clean_abstract_preserves_content(self):
        raw = "This is important content that should remain intact."
        cleaned = PMCEndpoint._clean_abstract(raw)

        assert "important content" in cleaned
        assert "remain intact" in cleaned


class TestFormatAPA:

    def test_format_apa_complete(self):
        """Test APA formatting with all fields present"""
        citation = PMCEndpoint._format_apa(
            authors=["Smith, J.", "Doe, A."],
            year="2024",
            title="Test Article",
            journal="Test Journal",
            volume="10",
            issue="3",
            pages="100–110",
            doi="10.1234/test.2024",
        )

        assert citation == (
            "Smith, J., & Doe, A. (2024). Test Article. "
            "Test Journal, 10(3), 100–110. https://doi.org/10.1234/test.2024"
        )

    def test_format_apa_single_author(self):
        citation = PMCEndpoint._format_apa(
            authors=["Jones, B."],
            year="2023",
            title="Solo Work",
            journal="Journal",
            volume="5",
            issue="1",
            pages="1–10",
            doi="10.1234/solo",
        )

        assert citation.startswith("Jones, B. (2023)")
        assert "&" not in citation

    def test_format_apa_no_authors(self):
        citation = PMCEndpoint._format_apa(
            authors=[],
            year="2024",
            title="No Author",
            journal="Journal",
            volume="1",
            issue="1",
            pages="1–5",
            doi="10.1234/anon",
        )

        assert citation.startswith(" (2024)")  # Empty author string

    def test_format_apa_no_issue(self):
        citation = PMCEndpoint._format_apa(
            authors=["Author, A."],
            year="2024",
            title="Test",
            journal="Journal",
            volume="10",
            issue="",
            pages="1–10",
            doi="10.1234/test",
        )

        assert "10()" not in citation  # Should not have empty parentheses
        assert ", 10," in citation  # Volume only

    def test_format_apa_no_doi(self):
        citation = PMCEndpoint._format_apa(
            authors=["Author, A."],
            year="2024",
            title="Test",
            journal="Journal",
            volume="10",
            issue="1",
            pages="1–10",
            doi="",
        )

        assert "https://doi.org/" not in citation
        assert citation.endswith("1–10. ")  # Ends with pages and space


class TestFetchPMCRecords:

    @patch.object(PMCEndpoint, '_parse_article')
    @patch("src.pmc_endpoint.ET.fromstring")
    @patch("src.pmc_endpoint.Entrez.efetch")
    @patch.object(PMCEndpoint, '_fetch_pmc_ids')
    def test_fetch_pmc_records_success(
        self, mock_fetch_ids, mock_efetch, mock_fromstring, mock_parse, mock_env_vars
    ):
        # Mock the ID fetching
        mock_fetch_ids.return_value = ["12345", "67890"]

        # Mock the efetch response
        mock_efetch_handle = MagicMock()
        mock_efetch_handle.read.side_effect = [
            "<article>Article 1</article>",
            "<article>Article 2</article>",
        ]
        mock_efetch_handle.close = MagicMock()
        mock_efetch.return_value = mock_efetch_handle

        # Mock XML parsing
        mock_root = MagicMock()
        mock_fromstring.return_value = mock_root

        # Mock parse_article
        mock_parse.side_effect = [
            {
                "pmcid": "12345",
                "apa_citation": "Citation 1",
                "abstract": "Abstract 1",
            },
            {
                "pmcid": "67890",
                "apa_citation": "Citation 2",
                "abstract": "Abstract 2",
            },
        ]

        records = PMCEndpoint.fetch_pmc_records("test query", retmax=2)

        assert len(records) == 2
        assert records[0]["pmcid"] == "12345"
        assert records[1]["pmcid"] == "67890"
        mock_fetch_ids.assert_called_once_with("test query", 2)
        assert mock_efetch.call_count == 2

    @patch.object(PMCEndpoint, '_fetch_pmc_ids')
    def test_fetch_pmc_records_no_results(self, mock_fetch_ids, mock_env_vars):
        mock_fetch_ids.return_value = []

        records = PMCEndpoint.fetch_pmc_records("nonexistent query")

        assert records == []

    @patch.object(PMCEndpoint, '_parse_article')
    @patch("src.pmc_endpoint.ET.fromstring")
    @patch("src.pmc_endpoint.Entrez.efetch")
    @patch.object(PMCEndpoint, '_fetch_pmc_ids')
    def test_fetch_pmc_records_parse_error_raises(
        self, mock_fetch_ids, mock_efetch, mock_fromstring, mock_parse, mock_env_vars
    ):
        mock_fetch_ids.return_value = ["12345"]

        mock_efetch_handle = MagicMock()
        mock_efetch_handle.read.return_value = "<article>Invalid</article>"
        mock_efetch_handle.close = MagicMock()
        mock_efetch.return_value = mock_efetch_handle

        mock_root = MagicMock()
        mock_fromstring.return_value = mock_root

        # Parse raises exception
        mock_parse.side_effect = Exception("Parse error")

        # Should raise the exception since there's no error handling
        with pytest.raises(Exception, match="Parse error"):
            PMCEndpoint.fetch_pmc_records("test")
