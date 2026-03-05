from typing import Dict, List

from src.medlit_agent.pmc_service.chroma_db import ChromaDB
from src.medlit_agent.pmc_service.embeddings_service import SBertEmbeddingsService
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint
from src.medlit_agent.pmc_service.xml_to_dict import XMLToDictConverter


class FullTextRetriever:
    """
    Get body from /pmc-articleset/article/body or //article/body fallback
    and convert section titles + paragraphs dict of chunks
    """

    def __init__(self):
        self.converter = XMLToDictConverter()
        self.endpoint = PMCEndpoint()
        self.db = ChromaDB()

    def retrieve_full_text(self, pmid: str) -> List[Dict[str, str]]:
        """
        Retrieve full text sections for a given PMID
        """
        xml_content = PMCEndpoint.fetch_pmcid_xml(pmid)
        sections = self.converter.convert(xml_content)
        return sections

    def store_full_text(self, pmid: str, sections: List[Dict[str, str]]):
        """
        Store full text sections in the database
        """
        self.db.add(pmid, sections)

    def query_full_text(self, query: str, n_results: int = 5) -> List[Dict[str, str]]:
        """
        Query the database for relevant sections based on the query
        """
        query_embedding = SBertEmbeddingsService.get_embedding(query)
        results = self.db.query(query_embedding, n_results)
        return results


if __name__ == "__main__":
    retriever = FullTextRetriever()
    pmid = "PMC6659366"
    sections = retriever.retrieve_full_text(pmid)
    retriever.store_full_text(pmid, sections)
    query_results = retriever.query_full_text("parkinsons", n_results=3)
    print(query_results)
