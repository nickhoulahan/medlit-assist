from __future__ import annotations

from typing import List, Self

from pydantic import BaseModel, Field


class ArticleDocument(BaseModel):
    pmcid: str = Field(..., description="PMC ID (e.g. '12345678' or 'PMC12345678')")
    citation: str = Field(..., description="APA-style citation")
    abstract: str = Field(..., description="Cleaned abstract text")


class LLMOutputModel(BaseModel):
    @classmethod
    def from_llm(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class ResearchSynthesis(LLMOutputModel):
    what_the_research_found: str = Field(
        ..., description="Main findings in simple, everyday language"
    )
    why_it_matters: str = Field(..., description="Practical implications")
    the_science_behind_it: str = Field(
        ..., description="Technical details explained accessibly"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of sources formatted like '(Title, https://pmc...)'",
    )

    def to_markdown(self) -> str:
        sources_block = "\n".join(f"- {source}" for source in self.sources)
        if not sources_block:
            sources_block = "- No source links were provided."
        return (
            "**What the research found:**\n\n"
            f"{self.what_the_research_found}\n\n"
            "**Why it matters:**\n\n"
            f"{self.why_it_matters}\n\n"
            "**The science behind it:**\n\n"
            f"{self.the_science_behind_it}\n\n"
            "**Sources:**\n"
            f"{sources_block}"
        )


class ArticleQAAnswer(LLMOutputModel):
    answer: str = Field(..., description="Answer in simple, everyday language")
    citations: List[str] = Field(
        default_factory=list,
        description="List of cited sources like 'Article 1 (PMC123...)' or '(Title, url)'",
    )

    def to_markdown(self) -> str:
        citations_block = "\n".join(f"- {citation}" for citation in self.citations)
        if not citations_block:
            citations_block = "- No citations were provided."
        return f"{self.answer}\n\n**Citations:**\n{citations_block}"

