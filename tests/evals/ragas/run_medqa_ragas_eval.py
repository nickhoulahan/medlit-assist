from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import random
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory

from src.medlit_agent.agent.agent import OllamaAgent
from src.medlit_agent.pmc_service.chroma_db import ChromaDB
from src.medlit_agent.pmc_service.embeddings_service import SBertEmbeddingsService
from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint
from src.medlit_agent.pmc_service.xml_to_dict import XMLToDictConverter
from src.medlit_agent.tools.tools import search_pubmed_central, tools
from tests.evals.ragas.ragas_metrics import build_ragas_metrics, score_metric


METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "answer_accuracy",
    "context_precision",
]

MAX_EVAL_REFERENCE_CHARS = 1200
REFERENCE_REQUIRED_METRICS = {"answer_correctness", "answer_accuracy"}


def _build_evaluator_llm(
    model_name: str,
    *,
    evaluator_max_tokens: int,
    ollama_base_url: str,
    ollama_api_key: str,
) -> Any:
    client = AsyncOpenAI(base_url=ollama_base_url, api_key=ollama_api_key)
    return llm_factory(
        model=model_name,
        provider="openai",
        client=client,
        max_tokens=evaluator_max_tokens,
        max_completion_tokens=evaluator_max_tokens,
        temperature=0.0,
    )


def _build_evaluator_embeddings(model_name: str) -> Any:
    return HuggingFaceEmbeddings(model=model_name)


def _resolve_eval_max_tokens(cli_value: int) -> int:
    if cli_value <= 0:
        return 800
    return cli_value


def _build_single_metric(
    *,
    metric_key: str,
    evaluator_model: str,
    evaluator_max_tokens: int,
    ollama_base_url: str,
    ollama_api_key: str,
    embedding_model: str,
    relevancy_strictness: int,
) -> Any:
    evaluator_llm = _build_evaluator_llm(
        evaluator_model,
        evaluator_max_tokens=evaluator_max_tokens,
        ollama_base_url=ollama_base_url,
        ollama_api_key=ollama_api_key,
    )
    evaluator_embeddings = _build_evaluator_embeddings(embedding_model)
    all_metrics = build_ragas_metrics(
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        relevancy_strictness=relevancy_strictness,
    )
    return all_metrics[metric_key]


def _default_csv_path() -> Path:
    return Path(__file__).parent / "medqa_data" / "medqa.csv"


def _report_dir() -> Path:
    return Path(__file__).parent / "reports"


def _load_medqa_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            qtype = (row.get("qtype") or "unknown").strip()
            if question and answer:
                rows.append(
                    {
                        "qtype": qtype,
                        "question": question,
                        "answer": answer,
                    }
                )
    return rows


def _subset_rows(
    rows: list[dict[str, str]],
    *,
    sample_size: int,
    seed: int,
    qtypes: list[str] | None,
) -> list[dict[str, str]]:
    filtered = rows
    if qtypes:
        allowed = {item.strip().lower() for item in qtypes if item.strip()}
        filtered = [row for row in rows if row["qtype"].lower() in allowed]

    if sample_size <= 0:
        return filtered

    rng = random.Random(seed)
    if sample_size >= len(filtered):
        return filtered

    return rng.sample(filtered, sample_size)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name.strip().lower()).strip("-")
    return slug or "unknown-model"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _prepare_contexts_for_eval(contexts: list[str]) -> list[str]:
    return contexts


def _clean_agent_response_for_eval(response: str) -> str:
    """
    Cleans agent response text for more reliable evaluation by RAGAS metrics.
    Otherwise, display characters and markdown influence metric scores
    """
    text = response or ""
    text = text.strip()
    if not text:
        return text

    # Drop tool-status chatter that can confuse judge models.
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("🔎 Searching PubMed Central"):
            continue
        if stripped.startswith("📚 Found "):
            continue
        if stripped.startswith("📄 Retrieving full text"):
            continue
        if stripped.startswith("💡 *Any other follow-up questions?"):
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()

    # Remove trailing follow-up separator blocks.
    cleaned = re.sub(r"\n---\s*$", "", cleaned).strip()

    # If the model emitted a JSON object, keep only parseable fields.
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                ordered_keys = [
                    "what_the_research_found",
                    "why_it_matters",
                    "the_science_behind_it",
                    "answer",
                ]
                chunks = [
                    str(payload[key]).strip()
                    for key in ordered_keys
                    if key in payload and str(payload[key]).strip()
                ]
                if chunks:
                    return "\n\n".join(chunks)
        except Exception:
            pass

    return cleaned


def _documents_to_contexts(documents: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for document in documents:
        abstract = str(document.get("abstract") or "").strip()
        if abstract:
            contexts.append(abstract)
    return contexts


def _retrieve_contexts_for_eval(question: str, pmc_max_results: int) -> list[str]:
    try:
        documents = search_pubmed_central.invoke(
            {"query": question, "max_results": pmc_max_results}
        )
        if not documents:
            return []
        return _documents_to_contexts(documents)
    except Exception:
        return []


async def _score_with_retries(
    *,
    score_fn,
    metric: Any,
    user_input: str,
    response: str,
    reference: str | None = None,
    retrieved_contexts: list[str] | None = None,
) -> float:
    candidate_responses = [
        response,
        response[:3000],
        response[:1500],
        response[:800],
        response[:400],
        response[:200],
    ]

    last_error: Exception | None = None
    for candidate in candidate_responses:
        try:
            if reference is None:
                return await score_fn(
                    metric,
                    user_input=user_input,
                    response=candidate,
                    retrieved_contexts=retrieved_contexts,
                )
            return await score_fn(
                metric,
                user_input=user_input,
                response=candidate,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


async def _generate_response(
    *,
    question: str,
    agent_model: str,
    agent_temperature: float,
    pmc_max_results: int,
) -> tuple[str, list[str]]:
    # Always use tools path for search mode.
    agent = OllamaAgent(
        model=agent_model,
        tools=tools,
        temperature=agent_temperature,
    )
    response = await agent.ainvoke(question)
    contexts = _retrieve_contexts_for_eval(question, pmc_max_results)
    return response, contexts


async def _generate_fulltext_response(
    *,
    question: str,
    pmcid: str,
    agent_model: str,
    agent_temperature: float,
    fulltext_n_results: int,
) -> tuple[str, list[str], str]:
    """Spin up and tear down a eval collection for evaluating RAG metric for full text QA."""
    collection_name = f"ragas_collection_{pmcid.lower()}_{uuid4().hex[:8]}"
    persist_dir = Path(tempfile.mkdtemp(prefix="ragas_fulltext_"))
    service = ChromaDB(collection_name=collection_name, persist_directory=persist_dir)

    try:
        xml_content = PMCEndpoint.fetch_pmcid_xml(pmcid)
        sections = XMLToDictConverter.convert(xml_content)
        service.add(pmcid=pmcid, texts=sections)

        query_embedding = SBertEmbeddingsService.get_embedding(question)
        metadatas = service.query(
            query_embedding=query_embedding,
            n_results=fulltext_n_results,
        )

        if not metadatas:
            raise ValueError(f"No chunks retrieved from temporary collection for {pmcid}")

        documents = [
            {
                "pmcid": str(item.get("pmcid") or pmcid),
                "title": str(item.get("title") or ""),
                "body": str(item.get("text") or ""),
            }
            for item in metadatas
            if str(item.get("text") or "").strip()
        ]
        if not documents:
            raise ValueError(f"No non-empty full-text chunks available for {pmcid}")

        contexts = [
            "\n\n".join(
                part
                for part in [doc.get("title", "").strip(), doc.get("body", "").strip()]
                if part
            )
            for doc in documents
        ]

        agent = OllamaAgent(
            model=agent_model,
            tools=tools,
            temperature=agent_temperature,
        )

        chunks: list[str] = []
        async for chunk in agent._stream_synthesis(user_input=question, documents=documents):
            chunks.append(chunk)
        return "".join(chunks).strip(), contexts, collection_name
    finally:
        try:
            service.client.delete_collection(name=collection_name)
        except Exception:
            pass
        shutil.rmtree(persist_dir, ignore_errors=True)


async def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv_path)
    qtypes = [item for item in args.qtypes.split(",") if item.strip()]

    eval_rows: list[dict[str, str]]
    if args.eval_mode == "search":
        all_rows = _load_medqa_rows(csv_path)
        eval_rows = _subset_rows(
            all_rows,
            sample_size=args.sample_size,
            seed=args.seed,
            qtypes=qtypes or None,
        )
        if not eval_rows:
            raise ValueError("No MedQA rows selected. Check --csv-path or --qtypes filter.")
    else:
        eval_rows = [
            {
                "qtype": "fulltext",
                "question": args.fulltext_question,
                "answer": "",
            }
        ]

    scored_rows: list[dict[str, Any]] = []

    tool_modes = ["tools"]

    for index, row in enumerate(eval_rows, start=1):
        question = row["question"]
        reference = row.get("answer", "")
        qtype = row["qtype"]

        for mode in tool_modes:
            sample_result: dict[str, Any] = {
                "index": index,
                "tool_mode": mode,
                "qtype": qtype,
                "question": question,
                "reference": reference or None,
                "status": "ok",
            }

            try:
                if args.eval_mode == "search":
                    raw_response, retrieved_contexts = await _generate_response(
                        question=question,
                        agent_model=args.agent_model,
                        agent_temperature=args.agent_temperature,
                        pmc_max_results=args.pmc_max_results,
                    )
                else:
                    raw_response, retrieved_contexts, collection_name = await _generate_fulltext_response(
                        question=question,
                        pmcid=args.fulltext_pmcid,
                        agent_model=args.agent_model,
                        agent_temperature=args.agent_temperature,
                        fulltext_n_results=args.fulltext_n_results,
                    )
                    sample_result["pmcid"] = args.fulltext_pmcid
                    sample_result["temporary_collection"] = collection_name
                response = _clean_agent_response_for_eval(raw_response)

                sample_result["response"] = response
                sample_result["raw_response"] = raw_response
                sample_result["retrieved_contexts"] = retrieved_contexts

            except Exception as exc:
                sample_result["status"] = "error"
                sample_result["error"] = str(exc)

            scored_rows.append(sample_result)

    base_config = {
        "csv_path": str(csv_path),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "qtypes": qtypes,
        "agent_model": args.agent_model,
        "agent_temperature": args.agent_temperature,
        "evaluator_model": args.evaluator_model,
        "evaluator_max_tokens": _resolve_eval_max_tokens(args.evaluator_max_tokens),
        "ollama_base_url": args.ollama_base_url,
        "embedding_service": "HuggingFaceEmbeddings",
        "embedding_model": args.embedding_model,
        "relevancy_strictness": args.relevancy_strictness,
        "metrics": METRIC_KEYS,
        "eval_mode": args.eval_mode,
        "tool_mode": "tools",
        "pmc_max_results": args.pmc_max_results,
        "fulltext_pmcid": args.fulltext_pmcid,
        "fulltext_question": args.fulltext_question,
        "fulltext_n_results": args.fulltext_n_results,
    }

    for metric_key in METRIC_KEYS:
        for row in scored_rows:
            if row.get("status") != "ok":
                row[f"{metric_key}_error"] = row.get("error") or "generation_failed"
                continue

            has_reference = bool(str(row.get("reference") or "").strip())
            if metric_key in REFERENCE_REQUIRED_METRICS and not has_reference:
                row[f"{metric_key}_skipped"] = "requires reference"
                continue

            eval_reference = (
                _truncate_text(str(row.get("reference") or ""), MAX_EVAL_REFERENCE_CHARS)
                if has_reference
                else None
            )
            eval_contexts = _prepare_contexts_for_eval(
                list(row.get("retrieved_contexts") or [])
            )

            metric: Any | None = None
            try:
                metric = _build_single_metric(
                    metric_key=metric_key,
                    evaluator_model=args.evaluator_model,
                    evaluator_max_tokens=_resolve_eval_max_tokens(
                        args.evaluator_max_tokens
                    ),
                    ollama_base_url=args.ollama_base_url,
                    ollama_api_key=args.ollama_api_key,
                    embedding_model=args.embedding_model,
                    relevancy_strictness=args.relevancy_strictness,
                )
                metric_score = await _score_with_retries(
                    score_fn=score_metric,
                    metric=metric,
                    user_input=str(row.get("question") or ""),
                    response=str(row.get("response") or ""),
                    reference=eval_reference,
                    retrieved_contexts=eval_contexts,
                )
                row[metric_key] = metric_score
            except Exception as exc:
                row[f"{metric_key}_error"] = str(exc)
            finally:
                if metric is not None:
                    del metric
                gc.collect()

    for row in scored_rows:
        if row.get("status") != "ok":
            continue
        metric_errors = {
            key: row.get(f"{key}_error")
            for key in METRIC_KEYS
            if row.get(f"{key}_error")
        }
        missing_scores = [
            key
            for key in METRIC_KEYS
            if row.get(key) is None and not row.get(f"{key}_skipped")
        ]
        if metric_errors or missing_scores:
            row["status"] = "error"
            if "error" not in row:
                row["error"] = "one or more metrics failed"
            if metric_errors:
                row["metric_errors"] = metric_errors

    summary = {
        "total": len(scored_rows),
        "ok": sum(1 for item in scored_rows if item.get("status") == "ok"),
        "errors": sum(1 for item in scored_rows if item.get("status") == "error"),
    }
    for metric_key in METRIC_KEYS:
        valid_scores = [
            float(item[metric_key])
            for item in scored_rows
            if item.get("status") == "ok" and item.get(metric_key) is not None
        ]
        summary[f"{metric_key}_mean"] = _safe_mean(valid_scores)

    summary_by_mode: dict[str, dict[str, float | int | None]] = {}
    for mode in sorted(set(item.get("tool_mode", "auto") for item in scored_rows)):
        mode_rows = [item for item in scored_rows if item.get("tool_mode") == mode]
        summary_by_mode[mode] = {
            "total": len(mode_rows),
            "ok": sum(1 for item in mode_rows if item.get("status") == "ok"),
            "errors": sum(1 for item in mode_rows if item.get("status") == "error"),
        }
        for metric_key in METRIC_KEYS:
            mode_scores = [
                float(item[metric_key])
                for item in mode_rows
                if item.get("status") == "ok" and item.get(metric_key) is not None
            ]
            summary_by_mode[mode][f"{metric_key}_mean"] = _safe_mean(mode_scores)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": base_config,
        "summary": summary,
        "summary_by_mode": summary_by_mode,
        "results": scored_rows,
    }


def _write_report(payload: dict[str, Any], suffix: str = "") -> Path:
    reports_dir = _report_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = payload.get("config", {})
    model_name = str(config.get("agent_model") or "unknown-model")
    model_slug = _slugify_model_name(model_name)
    json_path = reports_dir / f"ragas_medqa_report_{model_slug}{suffix}.json"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json_path


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAGAS v0.4 metrics evaluation against a "
            "subset of MedQA questions using the MedLit agent."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(_default_csv_path()),
        help="Path to medqa.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of MedQA rows to evaluate (0 means all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling rows",
    )
    parser.add_argument(
        "--qtypes",
        type=str,
        default="",
        help="Optional comma-separated qtype filter",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-oss:20b",
        choices=["gpt-oss:20b", "qwen3:8b", "gemma3:4b", "granite3.3:2b"],
        help="Ollama model used by the MedLit agent",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.0,
        help="Agent generation temperature",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-oss:20b",
        choices=["gpt-oss:20b", "qwen3:8b", "gemma3:4b", "granite3.3:2b"],
        help="Ollama model used by RAGAS evaluator",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "Optional convenience override to use the same Ollama model for "
            "both agent and evaluator"
        ),
    )
    parser.add_argument(
        "--evaluator-max-tokens",
        type=int,
        default=800,
        help=(
            "Maximum completion tokens for evaluator model calls "
            "used by RAGAS metrics"
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL for Ollama",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        default="ollama",
        help="API key used for OpenAI-compatible Ollama client, otherwise it fails",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name for RAGAS evaluator metrics",
    )
    parser.add_argument(
        "--relevancy-strictness",
        type=int,
        default=3,
        help="Number of synthetic questions generated per answer for relevancy",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["search", "fulltext"],
        default="search",
        help=(
            "Eval pipeline mode: search=MedQA query with search tool, "
            "fulltext=PMC full-text retrieval + chunking + temporary Chroma query"
        ),
    )
    parser.add_argument(
        "--pmc-max-results",
        type=int,
        default=3,
        help="Number of PMC records to retrieve in force-pmc mode",
    )
    parser.add_argument(
        "--fulltext-pmcid",
        type=str,
        default="PMC10923097",
        help="PMC ID used for fulltext eval mode",
    )
    parser.add_argument(
        "--fulltext-question",
        type=str,
        default="What role does nicotine play in preventing Parkinson's?",
        help="Question used for fulltext eval mode",
    )
    parser.add_argument(
        "--fulltext-n-results",
        type=int,
        default=5,
        help="Number of Chroma chunks retrieved for fulltext eval mode",
    )

    args = parser.parse_args()

    if args.model.strip():
        args.agent_model = args.model.strip()
        args.evaluator_model = args.model.strip()

    payload = await _run_eval(args)
    _write_report(payload, suffix=f"_{args.eval_mode}")


if __name__ == "__main__":
    asyncio.run(_amain())
