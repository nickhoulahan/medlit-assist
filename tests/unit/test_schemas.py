import pytest
from pydantic import ValidationError

from src.medlit_agent.schemas.schemas import (
    ArticleQAAnswer,
    ResearchSynthesis,
)


def test_parse_research_synthesis_from_json_string():
    payload = (
        '{"what_the_research_found": "A", "why_it_matters": "B", '
        '"the_science_behind_it": "C", "sources": ["(T, https://pmc...)"]}'
    )

    parsed = ResearchSynthesis.from_llm(payload)

    assert isinstance(parsed, ResearchSynthesis)
    assert parsed.what_the_research_found == "A"
    assert parsed.why_it_matters == "B"
    assert parsed.the_science_behind_it == "C"
    assert parsed.sources == ["(T, https://pmc...)"]


def test_parse_research_synthesis_from_fenced_json():
    payload = (
        "```json\n"
        '{"what_the_research_found": "A", "why_it_matters": "B", '
        '"the_science_behind_it": "C", "sources": []}'
        "\n```"
    )

    parsed = ResearchSynthesis.from_llm(payload)

    assert isinstance(parsed, ResearchSynthesis)
    assert parsed.sources == []


def test_parse_research_synthesis_accepts_legacy_science_key():
    payload = (
        '{"what_the_research_found": "A", "why_it_matters": "B", '
        '"the_science_below_it": "C", "sources": []}'
    )

    parsed = ResearchSynthesis.from_llm(payload)

    assert isinstance(parsed, ResearchSynthesis)
    assert parsed.the_science_behind_it == "C"


def test_parse_article_qa_answer_from_dict():
    parsed = ArticleQAAnswer.from_llm(
        '{"answer": "It helps", "citations": ["Article 1 (PMC123)"]}'
    )

    assert isinstance(parsed, ArticleQAAnswer)
    assert parsed.answer == "It helps"
    assert parsed.citations == ["Article 1 (PMC123)"]


def test_parse_article_qa_answer_raises_on_missing_answer():
    with pytest.raises(ValidationError):
        ArticleQAAnswer.from_llm('{"citations": []}')


def test_render_research_synthesis_markdown_contains_sections():
    model = ResearchSynthesis(
        what_the_research_found="Found A",
        why_it_matters="Matters because B",
        the_science_behind_it="Mechanism C",
        sources=["(Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC123)"]
    )

    rendered = model.to_markdown()

    assert "What the research found" in rendered
    assert "Why it matters" in rendered
    assert "The science behind it" in rendered
    assert "Sources" in rendered


def test_render_article_qa_markdown_contains_answer_and_citations():
    model = ArticleQAAnswer(answer="Short answer", citations=["Article 1 (PMC123)"])

    rendered = model.to_markdown()

    assert "Short answer" in rendered
    assert "Citations" in rendered
    assert "Article 1 (PMC123)" in rendered
