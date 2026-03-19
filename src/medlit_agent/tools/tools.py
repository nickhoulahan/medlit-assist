from typing import Dict, List

from langchain_core.tools import tool
from langsmith import traceable

from src.medlit_agent.pmc_service.full_text_retriever import FullTextRetriever
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint


@tool
def search_pubmed_central(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search PubMed Central for biomedical research articles.

    Args:
        query: The search query (e.g., "cancer therapy", "diabetes treatment")
        max_results: Maximum number of articles to return (default: 3)

    Returns:
        List of articles with PMC ID, APA citation, and abstract
    """
    try:
        pmc_results = PMCEndpoint.fetch_pmc_records(query, retmax=max_results)

        documents = []
        for result in pmc_results:
            documents.append(
                {
                    "pmcid": result["pmcid"],
                    "citation": result["apa_citation"],
                    "abstract": result["abstract"],
                }
            )

        return documents
    except Exception as e:
        raise Exception(f"Error searching PubMed Central: {str(e)}")


@tool
def retrieve_full_text(pmcid: str) -> List[Dict[str, str]]:
    """Retrieve full text sections for a given PMC ID to answer questions on article-spefic user queries.
    Args:
        pmcid: The PMC ID of the article (e.g., "PMC1013555")
        Returns: List of full text sections with title and body
    """
    retriever = FullTextRetriever()
    sections = retriever.retrieve_full_text(pmcid)
    return sections


# Export tools list for easy import
tools = [search_pubmed_central, retrieve_full_text]
