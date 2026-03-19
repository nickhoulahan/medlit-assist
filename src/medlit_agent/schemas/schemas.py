from __future__ import annotations

import re
from typing import List, Self

from pydantic import AliasChoices, BaseModel, Field


class ArticleDocument(BaseModel):
    pmcid: str = Field(..., description="PMC ID (e.g. '12345678' or 'PMC12345678')")
    citation: str = Field(..., description="APA-style citation")
    abstract: str = Field(..., description="Cleaned abstract text")


class LLMOutputModel(BaseModel):
    @staticmethod
    def _extract_json_payload(payload: str) -> str | None:
        """
            Extract JSON object from the input string, handling cases where the JSON may be fenced or embedded within other text.
            This private method guards against an I was noticing where the JSON may be wrapped in additional text or formatting.
        """
        text = payload.strip()
        if not text:
            return None

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            return fenced_match.group(1).strip()

        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            return object_match.group(0).strip()

        return None

    @classmethod
    def from_llm(cls, payload: str) -> Self:
        try:
            return cls.model_validate_json(payload)
        except Exception:
            extracted = cls._extract_json_payload(payload)
            if not extracted or extracted == payload.strip():
                raise
            return cls.model_validate_json(extracted)


class ResearchSynthesis(LLMOutputModel):
    what_the_research_found: str = Field(
        ..., description="Main findings in simple, everyday language"
    )
    why_it_matters: str = Field(..., description="Practical implications")
    the_science_behind_it: str = Field(
        ...,
        validation_alias=AliasChoices("the_science_behind_it", "the_science_below_it"),
        description="Technical details explained accessibly",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of sources formatted like '(Title, https://pmc...)'",
    )

    def to_markdown(self, include_sources: bool = True) -> str:
        base = (
            "**What the research found:**\n\n"
            f"{self.what_the_research_found}\n\n"
            "**Why it matters:**\n\n"
            f"{self.why_it_matters}\n\n"
            "**The science behind it:**\n\n"
            f"{self.the_science_behind_it}"
        )
        if not include_sources:
            return base

        sources_block = "\n".join(f"- {source}" for source in self.sources)
        if not sources_block:
            sources_block = "- No source links were provided."
        return f"{base}\n\n**Sources:**\n{sources_block}"


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

