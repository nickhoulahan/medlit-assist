from langchain_core.messages import HumanMessage, SystemMessage

from src.medlit_agent.graph.langgraph_workflow import (
    build_qa_messages,
    build_synthesis_messages,
)


def _sample_documents():
    return [
        {
            "pmcid": "PMC0001",
            "citation": "Smith, J. (2025). Study. Journal.",
            "abstract": "Abstract one.",
        },
        {
            "pmcid": "PMC0002",
            "citation": "Doe, A. (2024). Another Study. Journal.",
            "abstract": "Abstract two.",
        },
    ]


def test_build_synthesis_messages_produces_prompt_sequence():
    messages = build_synthesis_messages("diabetes treatment", _sample_documents())

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "What the research found" in messages[0].content
    assert "diabetes treatment" in messages[1].content


def test_build_qa_messages_includes_contextual_instructions():
    messages = build_qa_messages("why aspirin works", _sample_documents())

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Answer the user's question" in messages[0].content
    assert "Research articles" in messages[1].content
    assert "why aspirin works" in messages[1].content
