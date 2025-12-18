import os
import re
from xml.etree import ElementTree as ET

from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()


class PMCEndpoint:
    # email and api key allow for increased rate limits with NCBI Entrez

    endpoint = Entrez
    endpoint.email = os.getenv("EMAIL")
    endpoint.tool = "pmc_apa_abstract_fetcher"
    endpoint.api_key = os.getenv("PMC_API_KEY")

    @classmethod
    def _fetch_pmc_ids(cls, query, retmax=5):
        """Search for PMC IDs matching the query."""
        handle = cls.endpoint.esearch(db="pmc", term=query, retmax=retmax)
        record = cls.endpoint.read(handle)
        handle.close()
        return record.get("IdList", [])

    @classmethod
    def fetch_pmc_records(cls, query, retmax=5):
        """Use private methods to fetch and parse PMC XML records."""
        pmc_ids = cls._fetch_pmc_ids(query, retmax)
        articles = []

        for pmcid in pmc_ids:
            handle = cls.endpoint.efetch(
                db="pmc", id=pmcid, rettype="full", retmode="xml"
            )
            xml_data = handle.read()
            handle.close()

            root = ET.fromstring(xml_data)
            articles.append(cls._parse_article(root, pmcid))

        return articles

    @staticmethod
    def _parse_article(root, pmcid):
        """XML needs to be parsed to extract needed fields for an APA citation."""
        def find_text(path):
            """Helper method to find text content."""
            el = root.find(path)
            return el.text.strip() if el is not None and el.text else ""

        title = find_text(".//article-title")

        # get author info from different tags
        authors = []
        for contrib in root.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.findtext(".//surname", "")
            given = contrib.findtext(".//given-names", "")
            if surname and given:
                authors.append(f"{surname}, {given[0]}.")

        year = ""
        for pd in root.findall(".//pub-date"):
            if pd.attrib.get("pub-type") in ("epub", "ppub"):
                y = pd.findtext("year")
                if y:
                    year = y
                    break

        journal = find_text(".//journal-title")
        journal = re.sub(r"\s*\|\s*", " ", journal)

        volume = find_text(".//volume")
        issue = find_text(".//issue")

        fpage = find_text(".//fpage")
        lpage = find_text(".//lpage")
        pages = f"{fpage}\u2013{lpage}" if fpage and lpage else ""

        doi = ""
        for aid in root.findall(".//article-id"):
            if aid.attrib.get("pub-id-type") == "doi" and aid.text:
                doi = aid.text.replace("https://doi.org/", "").strip()
                break

        # abstract needs to be cleaned and resassembled
        raw_abstract = ""
        abstract_node = root.find(".//abstract")

        if abstract_node is not None:
            # Extract text from each paragraph element to preserve structure
            paragraphs = []
            for p in abstract_node.findall(".//p"):
                para_text = " ".join(
                    text.strip() for text in p.itertext() if text.strip()
                )
                if para_text:
                    paragraphs.append(para_text)

            # If no <p> tags, fall back to all text
            if paragraphs:
                raw_abstract = "\n\n".join(paragraphs)
            else:
                raw_abstract = " ".join(
                    text.strip() for text in abstract_node.itertext() if text.strip()
                )

        abstract = PMCEndpoint._clean_abstract(raw_abstract)

        # Generate APA citation
        apa_citation = PMCEndpoint._format_apa(
            authors, year, title, journal, volume, issue, pages, doi
        )

        return {"pmcid": pmcid, "apa_citation": apa_citation, "abstract": abstract}


    @staticmethod
    def _clean_abstract(raw_abstract: str) -> str:
        text = raw_abstract

        # Normalize whitespace
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Fix broken chemical notation (SO 2 → SO₂)
        text = re.sub(r"SO\s*2", "SO₂", text)

        # Need to add section headers if they exist
        headers = [
            "Objective",
            "Impact Statement",
            "Introduction",
            "Methods",
            "Results",
            "Conclusion",
        ]

        for h in headers:
            text = re.sub(rf"\b{h}:\s*", f"\n\n{h}: ", text, flags=re.IGNORECASE)

        text = text.strip()

        return text

    @staticmethod
    def _format_apa(authors, year, title, journal, volume, issue, pages, doi):

        if not authors:
            author_str = ""
        elif len(authors) == 1:
            author_str = authors[0]
        else:
            author_str = ", ".join(authors[:-1]) + ", & " + authors[-1]

        vol_issue = f"{volume}({issue})" if issue else volume
        doi_url = f"https://doi.org/{doi}" if doi else ""

        citation = (
            f"{author_str} ({year}). {title}. "
            f"{journal}, {vol_issue}, {pages}. {doi_url}"
        )

        return citation
