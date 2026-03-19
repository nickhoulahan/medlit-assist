import pytest

from src.medlit_agent.agent.agent import OllamaAgent

pytestmark = pytest.mark.asyncio


def _model_name() -> str:
    import os

    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


async def test_live_ollama_structured_qa_output():
    agent = OllamaAgent(model=_model_name(), temperature=0.0)
    agent.documents = [
        {
            "pmcid": "PMC123456",
            "citation": "Smith, J. (2025). Sample study. Example Journal.",
            "abstract": "The study reports a reduction in symptom severity in the intervention group.",
        }
    ]

    chunks = []
    async for chunk in agent.astream("What did this study find?"):
        chunks.append(chunk)

    output = "".join(chunks)

    assert output.strip()
    assert "Citations:" in output


async def test_live_ollama_structured_synthesis_output():
    agent = OllamaAgent(model=_model_name(), temperature=0.0)
    documents = [
        {
            "pmcid": "PMC123456",
            "citation": "Smith, J. (2025). Sample study. Example Journal.",
            "abstract": "The intervention was associated with improved outcomes compared with control.",
        }
    ]

    chunks = []
    async for chunk in agent._stream_synthesis(
        user_input="intervention outcomes", documents=documents
    ):
        chunks.append(chunk)

    output = "".join(chunks)

    assert output.strip()
    assert "What the research found" in output
    assert "Why it matters" in output
    assert "The science behind it" in output
