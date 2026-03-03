from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ArticleDocument(BaseModel):
    pmcid: str = Field(..., description="PMC ID (e.g. '12345678' or 'PMC12345678')")
    citation: str = Field(..., description="APA-style citation")
    abstract: str = Field(..., description="Cleaned abstract text")


class ResearchSynthesis(BaseModel):
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


class ArticleQAAnswer(BaseModel):
    answer: str = Field(..., description="Answer in simple, everyday language")
    citations: List[str] = Field(
        default_factory=list,
        description="List of cited sources like 'Article 1 (PMC123...)' or '(Title, url)'",
    )
