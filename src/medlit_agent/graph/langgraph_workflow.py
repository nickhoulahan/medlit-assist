from __future__ import annotations

from typing import Annotated, List, Mapping, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired

from src.medlit_agent.graph.langgraph_helpers import (
    build_documents_context,
    build_qa_prompts,
    build_synthesis_prompts,
)

# Types for clarity when wiring the graph, accepts either dicts or Pydantic models.
DocumentPayload = Mapping[str, str]


class MedlitChatState(TypedDict):
    user_input: str
    documents: List[DocumentPayload]
    documents_context: NotRequired[str]
    system_prompt: NotRequired[str]
    human_prompt: NotRequired[str]
    messages: Annotated[List[AnyMessage], add_messages]


def _format_documents_context(state: MedlitChatState) -> dict[str, str]:
    return {"documents_context": build_documents_context(state["documents"])}


def _build_synthesis_prompt(state: MedlitChatState) -> dict[str, str]:
    system, human = build_synthesis_prompts(
        state["user_input"], state["documents_context"]
    )
    return {"system_prompt": system, "human_prompt": human}


def _build_qa_prompt(state: MedlitChatState) -> dict[str, str]:
    system, human = build_qa_prompts(state["user_input"], state["documents_context"])
    return {"system_prompt": system, "human_prompt": human}


def _compose_messages(state: MedlitChatState) -> dict[str, list[AnyMessage]]:
    return {
        "messages": [
            SystemMessage(content=state["system_prompt"]),
            HumanMessage(content=state["human_prompt"]),
        ]
    }


def _build_prompt_graph(
    prompt_node,
) -> "CompiledStateGraph[MedlitChatState, None, MedlitChatState, MedlitChatState]":
    graph = StateGraph(MedlitChatState)
    graph.add_node("format_documents", _format_documents_context)
    graph.add_node("prepare_prompts", prompt_node)
    graph.add_node("compose_messages", _compose_messages)
    graph.set_entry_point("format_documents")
    graph.add_edge("format_documents", "prepare_prompts")
    graph.add_edge("prepare_prompts", "compose_messages")
    graph.set_finish_point("compose_messages")
    return graph.compile()


_SYNTHESIS_GRAPH = _build_prompt_graph(_build_synthesis_prompt)
_QA_GRAPH = _build_prompt_graph(_build_qa_prompt)


def _run_graph(
    compiled_graph,
    user_input: str,
    documents: List[DocumentPayload],
) -> List[AnyMessage]:
    state = {"user_input": user_input, "documents": documents, "messages": []}
    output = compiled_graph.invoke(state)
    return output["messages"]


def build_synthesis_messages(
    user_input: str, documents: List[DocumentPayload]
) -> List[AnyMessage]:
    return _run_graph(_SYNTHESIS_GRAPH, user_input, documents)


def build_qa_messages(
    user_input: str, documents: List[DocumentPayload]
) -> List[AnyMessage]:
    return _run_graph(_QA_GRAPH, user_input, documents)
