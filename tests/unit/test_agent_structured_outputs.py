from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.medlit_agent.agent.agent import OllamaAgent


def _stream_chunks(text: str, size: int = 80):
    async def _gen():
        for idx in range(0, len(text), size):
            yield SimpleNamespace(content=text[idx : idx + size])

    return _gen()


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

    mock_llm.ainvoke = AsyncMock(return_value=first_response)
    mock_llm.astream = MagicMock(
        return_value=_stream_chunks(synthesis_response.content)
    )

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
    assert "Result A" in output
    assert agent.last_validated_response is not None
    assert "What the research found" in agent.last_validated_response
    assert "PMC123456" in agent.last_validated_response


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

    mock_llm.ainvoke = AsyncMock(return_value=first_response)
    mock_llm.astream = MagicMock(return_value=_stream_chunks(qa_response.content))

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

    assert "Answer" in output
    assert "It may reduce risk." in output
    assert agent.last_validated_response is not None
    assert "Citations" in agent.last_validated_response


@pytest.mark.asyncio
@patch("src.medlit_agent.agent.agent.ChatOllama")
async def test_full_text_tool_output_omits_sources_section(mock_ollama):
    mock_llm = MagicMock()
    mock_ollama.return_value = mock_llm

    mock_tool = MagicMock()
    mock_tool.name = "retrieve_full_text"
    mock_tool.description = "Retrieve full article sections"
    mock_tool.invoke.return_value = [
        {
            "title": "Results",
            "body": "Detailed findings about treatment effects.",
        }
    ]

    first_response = SimpleNamespace(
        tool_calls=[{"name": "retrieve_full_text", "args": {"pmcid": "PMC123456"}}],
        content="",
    )
    synthesis_response = SimpleNamespace(
        tool_calls=[],
        content=(
            '{"what_the_research_found": "Result A", '
            '"why_it_matters": "Reason B", '
            '"the_science_behind_it": "Mechanism C"}'
        ),
    )

    mock_llm.ainvoke = AsyncMock(return_value=first_response)
    mock_llm.astream = MagicMock(
        return_value=_stream_chunks(synthesis_response.content)
    )

    agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
    agent.llm = mock_llm
    agent.llm_with_tools = mock_llm

    output_chunks = []
    async for chunk in agent.astream("What did this article find?"):
        output_chunks.append(chunk)

    output = "".join(output_chunks)

    assert "What the research found" in output
    assert "Why it matters" in output
    assert "The science behind it" in output
    assert "Sources" not in output
    assert agent.last_validated_response is not None
    assert "Sources" not in agent.last_validated_response


@pytest.mark.asyncio
@patch("src.medlit_agent.agent.agent.ChatOllama")
async def test_structured_synthesis_streams_in_multiple_chunks(mock_ollama):
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
            '{"what_the_research_found": "' + ("A" * 220) + '", '
            '"why_it_matters": "' + ("B" * 220) + '", '
            '"the_science_behind_it": "' + ("C" * 220) + '", '
            '"sources": ["(Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC123456)"]}'
        ),
    )

    mock_llm.ainvoke = AsyncMock(return_value=first_response)
    mock_llm.astream = MagicMock(
        return_value=_stream_chunks(synthesis_response.content, size=22)
    )

    agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
    agent.llm = mock_llm
    agent.llm_with_tools = mock_llm

    output_chunks = []
    async for chunk in agent.astream("What does research say about diabetes?"):
        output_chunks.append(chunk)

    # Two status messages are emitted before synthesis; ensure synthesis itself is chunked too.
    assert len(output_chunks) > 3


@pytest.mark.asyncio
@patch("src.medlit_agent.agent.agent.ChatOllama")
async def test_structured_qa_streams_in_multiple_chunks(mock_ollama):
    mock_llm = MagicMock()
    mock_ollama.return_value = mock_llm

    first_response = SimpleNamespace(tool_calls=[], content="")
    qa_response = SimpleNamespace(
        tool_calls=[],
        content=(
            '{"answer": "'
            + ("Long answer segment. " * 30)
            + '", "citations": ["Article 1 (PMC123456)"]}'
        ),
    )

    mock_llm.ainvoke = AsyncMock(return_value=first_response)
    mock_llm.astream = MagicMock(
        return_value=_stream_chunks(qa_response.content, size=20)
    )

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

    # QA path has no status preamble, so >1 confirms chunked streaming behavior.
    assert len(output_chunks) > 1
