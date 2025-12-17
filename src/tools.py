from typing import Dict, List

from langchain_core.tools import tool

from src.pmc_endpoint import PMCEndpoint


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


# Export tools list for easy import
tools = [search_pubmed_central]
