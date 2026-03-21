from typing import Dict, List

from langchain_core.tools import tool

from src.medlit_agent.pmc_service.full_text_retriever import FullTextRetriever
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint


@tool
def search_pubmed_central(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search PubMed Central for biomedical research articles.

    Args:
        query: The search query (e.g., "cancer therapy", "diabetes treatment")
        max_results: Maximum number of articles to return (default: 5)

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
    """Retrieve full text sections for a given PMC ID article in response to a 
    user's query about follow questions about a specific article. For example, if the
    user is following up on a previous search and wants to know more details about a specific article.
    Exmple: How does the second article in your response discuss medication side effects?
    Args:
        pmcid: The PMC ID of the article (e.g., "PMC1013555")
        Returns: List of full text sections with title and body
    """
    retriever = FullTextRetriever()
    sections = retriever.retrieve_full_text(pmcid)
    return sections


# Export tools list for easy import
tools = [search_pubmed_central, retrieve_full_text]
