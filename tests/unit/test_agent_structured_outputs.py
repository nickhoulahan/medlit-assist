from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.medlit_agent.agent.agent import OllamaAgent


@pytest.mark.asyncio
@patch("src.medlit_agent.agent.agent.ChatOllama")
async def test_structured_synthesis_output_is_rendered(mock_ollama):
    mock_llm = MagicMock()
    mock_ollama.return_value = mock_llm

    mock_tool = MagicMock()
    mock_tool.name = "search_pubmed_central"
    mock_tool.description = "Search for articles"
    mock_tool.invoke.return_value = [
        {
            "pmcid": "PMC123456",
            "citation": "Author. (2025). Title.",
            "abstract": "This is an abstract.",
        }
    ]

    first_response = SimpleNamespace(
        tool_calls=[
            {
                "name": "search_pubmed_central",
                "args": {"query": "diabetes", "max_results": 3},
            }
        ],
        content="",
    )
    synthesis_response = SimpleNamespace(
        tool_calls=[],
        content=(
            '{"what_the_research_found": "Result A", '
            '"why_it_matters": "Reason B", '
            '"the_science_behind_it": "Mechanism C", '
            '"sources": ["(Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC123456)"]}'
        ),
    )

    mock_llm.ainvoke = AsyncMock(side_effect=[first_response, synthesis_response])

    agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
    agent.llm = mock_llm
    agent.llm_with_tools = mock_llm

    output_chunks = []
    async for chunk in agent.astream("What does research say about diabetes?"):
        output_chunks.append(chunk)

    output = "".join(output_chunks)

    assert "What the research found" in output
    assert "Why it matters" in output
    assert "The science behind it" in output
    assert "PMC123456" in output


@pytest.mark.asyncio
@patch("src.medlit_agent.agent.agent.ChatOllama")
async def test_structured_qa_output_is_rendered(mock_ollama):
    mock_llm = MagicMock()
    mock_ollama.return_value = mock_llm

    first_response = SimpleNamespace(tool_calls=[], content="")
    qa_response = SimpleNamespace(
        tool_calls=[],
        content='{"answer": "It may reduce risk.", "citations": ["Article 1 (PMC123456)"]}',
    )

    mock_llm.ainvoke = AsyncMock(side_effect=[first_response, qa_response])

    agent = OllamaAgent(model="gpt-oss:20b")
    agent.llm = mock_llm
    agent.llm_with_tools = mock_llm
    agent.documents = [
        {
            "pmcid": "PMC123456",
            "citation": "Author. (2025). Title.",
            "abstract": "This is an abstract.",
        }
    ]

    output_chunks = []
    async for chunk in agent.astream("Does it help?"):
        output_chunks.append(chunk)

    output = "".join(output_chunks)

    assert "It may reduce risk." in output
    assert "Citations" in output
    assert "PMC123456" in output
