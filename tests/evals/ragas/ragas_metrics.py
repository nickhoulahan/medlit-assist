from __future__ import annotations

import inspect
from typing import Any

from ragas.metrics.collections import (
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    Faithfulness,
)


def _extract_metric_value(result: Any) -> float:
    if hasattr(result, "value"):
        return float(result.value)
    return float(result)


def _build_metric(metric_cls: Any, *, llm: Any, embeddings: Any, strictness: int) -> Any:
    """Supplies RAGAS metric with supported kwargs and initializes it."""
    init_signature = inspect.signature(metric_cls.__init__)
    supported_args = set(init_signature.parameters.keys())

    kwargs: dict[str, Any] = {}
    if "llm" in supported_args:
        kwargs["llm"] = llm
    if "embeddings" in supported_args:
        kwargs["embeddings"] = embeddings
    if "strictness" in supported_args:
        kwargs["strictness"] = strictness

    return metric_cls(**kwargs)


def build_ragas_metrics(
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    relevancy_strictness: int = 3,
) -> dict[str, Any]:
    return {
        "faithfulness": _build_metric(
            Faithfulness,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=relevancy_strictness,
        ),
        "answer_relevancy": _build_metric(
            AnswerRelevancy,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=relevancy_strictness,
        ),
        "answer_correctness": _build_metric(
            AnswerCorrectness,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=relevancy_strictness,
        ),
        "answer_accuracy": _build_metric(
            AnswerAccuracy,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=relevancy_strictness,
        ),
        "context_precision": _build_metric(
            ContextPrecision,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=relevancy_strictness,
        ),
    }


async def score_metric(
    metric: Any,
    user_input: str,
    response: str,
    reference: str | None = None,
    retrieved_contexts: list[str] | None = None,
) -> float:
    """Runs a RAGAS metric's ascore method with supported kwargs and returns result"""
    ascore_signature = inspect.signature(metric.ascore)
    supported_args = set(ascore_signature.parameters.keys())

    payload = {
        "user_input": user_input,
        "response": response,
        "reference": reference,
        "retrieved_contexts": retrieved_contexts,
    }

    kwargs = {
        key: value
        for key, value in payload.items()
        if key in supported_args and value is not None
    }

    result = await metric.ascore(**kwargs)
    return _extract_metric_value(result)
