import os
from datetime import datetime, timezone
import json
from pathlib import Path
import slugify

import pytest

from src.medlit_agent.agent.agent import OllamaAgent
from src.medlit_agent.schemas.schemas import ResearchSynthesis
from src.medlit_agent.tools.tools import search_pubmed_central, tools
from tests.evals.readability.readability import ReadabilityCalculator


pytestmark = pytest.mark.asyncio


def _model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "gpt-oss:20b")


def _report_dir() -> Path:
    return Path(__file__).parent / "reports"


def _review_files_dir() -> Path:
    return Path(__file__).parent / "review_files"


def _prompt_variants() -> list[tuple[str, list[tuple[str, str]]]]:
    questions = [
        "Can diet help with type 2 diabetes?",
        "Does smoking prevent Parkinson's disease?",
        "What drug preventions are there for Alzheimer's disease?",
    ]
    return [
        (
            question,
            [
                (
                    "baseline",
                    f"{question}",
                ),
                (
                    "plain_language",
                    f"{question} Explain this in simple, everyday language for non-experts. Avoid technical terms and jargon.",
                ),
                (
                    "very_simple",
                    f"{question} Use short sentences, common words, and avoid medical jargon. A middle school reading level is ideal. Explain any necessary technical terms in simple language.",
                ),
            ],
        )
        for question in questions
    ]


def _build_review_text(report_payload: dict) -> str:
    lines: list[str] = []
    lines.append("LLM Readability Review Report")
    lines.append("")
    lines.append(f"Generated: {report_payload.get('generated_at', '')}")
    lines.append(f"Model: {report_payload.get('model', '')}")
    lines.append(f"PMC Query: {report_payload.get('pmc_query', '')}")
    lines.append(f"PMC Results: {report_payload.get('pmc_results', '')}")
    lines.append("")
    lines.append("=" * 80)

    for idx, row in enumerate(report_payload.get("results", []), start=1):
        lines.append("")
        lines.append(f"Prompt {idx}: {row.get('prompt_label', 'unknown')}")
        lines.append(f"Status: {row.get('status', '')}")

        if row.get("status") == "ok":
            lines.append(
                "Scores: "
                f"FRE={row.get('flesch_reading_ease', 0):.2f}, "
                f"FKG={row.get('flesch_kincaid_grade', 0):.2f}, "
                f"Words={row.get('word_count', 0)}, "
                f"Sentences={row.get('sentence_count', 0)}"
            )
            lines.append("")
            lines.append("Prompt:")
            lines.append(str(row.get("prompt", "")))
            lines.append("")
            lines.append("Output:")
            lines.append(str(row.get("output", "")))
        lines.append("")
        lines.append("-" * 80)

    lines.append("")
    return "\n".join(lines)


def _ensure_render_output(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        return output

    try:
        return ResearchSynthesis.from_llm(stripped).to_markdown()
    except Exception:
        return output


async def test_live_llm_readability_report():
    agent = OllamaAgent(model=_model_name(), tools=tools, temperature=0.0)

    reports_dir = _report_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    review_dir = _review_files_dir()
    review_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    for question, variants in _prompt_variants():
        documents = search_pubmed_central.invoke(
            {"query": question, "max_results": 3}
        )
        if not documents:
            print(f"READABILITY {question} skipped=no_pmc_results")
            continue

        rows = []

        for label, user_input in variants:
            chunks = []
            async for chunk in agent._stream_synthesis(
                user_input=user_input,
                documents=documents,
            ):
                chunks.append(chunk)

            output = "".join(chunks).strip()
            if not output:
                print(f"READABILITY {question} [{label}] status=no_output")
                rows.append({"prompt_label": label, "status": "no_output", "prompt": user_input})
                continue

            output = _ensure_render_output(output)

            scores = ReadabilityCalculator.score_readability(output)
            rows.append(
                {
                    "prompt_label": label,
                    "status": "ok",
                    "flesch_reading_ease": scores.flesch_reading_ease,
                    "flesch_kincaid_grade": scores.flesch_kincaid_grade,
                    "word_count": scores.word_count,
                    "sentence_count": scores.sentence_count,
                    "prompt": user_input,
                    "output": output,
                }
            )

        ok_rows = [row for row in rows if row.get("status") == "ok"]
        if not ok_rows:
            print(f"READABILITY {question} skipped=no_readable_outputs")
            continue

        report_payload = {
            "generated_at": generated_at,
            "model": _model_name(),
            "question": question,
            "pmc_query": question,
            "pmc_results": len(documents),
            "results": rows,
        }

        question_slug = slugify.slugify(question)

        report_path = reports_dir / f"readability_report_{question_slug}.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        review_txt_path = review_dir / f"readability_review_{question_slug}.txt"
        review_text = _build_review_text(report_payload)
        review_txt_path.write_text(review_text, encoding="utf-8")
