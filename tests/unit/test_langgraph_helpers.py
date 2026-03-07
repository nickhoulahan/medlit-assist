from __future__ import annotations

from types import SimpleNamespace

from src.medlit_agent.graph.langgraph_helpers import (
    build_documents_context,
    build_qa_prompts,
    build_synthesis_prompts,
    build_tool_descriptions,
)


def test_build_tool_descriptions_returns_empty_when_no_tools():
    assert build_tool_descriptions({}) == ""


def test_build_tool_descriptions_includes_tool_info():
    tools = {"search": SimpleNamespace(description="Finds articles")}

    description = build_tool_descriptions(tools)

    assert "Available tools" in description
    assert "- search: Finds articles" in description


def test_build_documents_context_formats_articles():
    documents = [
        {"pmcid": "PMC123", "citation": "Author. (2025)", "abstract": "Summary."},
        {"pmcid": "PMC456", "citation": "Other. (2024)", "abstract": "More info."},
    ]

    context = build_documents_context(documents)

    assert "Article 1 (PMC ID: PMC123)" in context
    assert "Article 2 (PMC ID: PMC456)" in context


def test_build_synthesis_prompts_include_guidance():
    system, human = build_synthesis_prompts("diabetes", "context here")

    assert "What the research found" in system
    assert "Why it matters" in system
    assert "The science behind it" in system
    assert "Based on these research articles" in human
    assert "Research Articles" in human
    assert "diabetes" in human


def test_build_qa_prompts_references_context_and_question():
    system, human = build_qa_prompts("hypertension", "doc context")

    assert "Answer the user's question" in system
    assert "Research articles" in human
    assert "User question" in human
    assert "hypertension" in human
