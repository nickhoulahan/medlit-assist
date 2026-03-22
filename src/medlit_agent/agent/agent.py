from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.medlit_agent.graph.langgraph_helpers import build_tool_descriptions
from src.medlit_agent.graph.langgraph_workflow import (
    build_qa_messages,
    build_synthesis_messages,
)
from src.medlit_agent.schemas.schemas import (
    ArticleQAAnswer,
    ResearchSynthesis,
)


class OllamaAgent:

    def __init__(
        self,
        model: str,
        tools: List = None,
        temperature: float = 0.0,
    ):
        """
        Ollama agent with tools.

        Args:
            model: the Ollama model id to use
            tools: list of LangChain tools to provide to the agent
            temperature: model temperature (0-1), default 0.0
        """
        self.model = model
        self.tools_list = tools or []
        self.tools = {tool.name: tool for tool in self.tools_list}
        self.documents = []  # for storing fetched documents
        self.last_validated_response: Optional[str] = None

        # ChatOllama instance
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

        # Bind tools to llm
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools_list)
        except Exception:
            # Fallback to base llm if tool binding fails to keep agent usable
            self.llm_with_tools = self.llm

    def _extract_tool_args(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if tool_name == "search_pubmed_central":
            query = tool_args.get("query", "")
            max_results = tool_args.get("max_results", 3)
            return {"query": query, "max_results": max_results}

        if tool_name == "retrieve_full_text":
            pmcid = tool_args.get("pmcid", "")
            return {"pmcid": pmcid}

        return tool_args

    def _run_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Executes a single tool call and normalize storage for follow-up Q&A."""
        tool = self.tools[tool_name]
        result = tool.invoke(tool_args)

        # Store documents in memory for follow-up Q&A.
        # (Keeping as list[dict] to preserve existing app/tests expectations.)
        # different from chroma vector DB caching.
        self.documents = result
        return result

    @staticmethod
    def _is_full_text_unavailable_error(exc: Exception) -> bool:
        msg = str(exc).casefold()
        return "no <body> element found" in msg or "cannot extract full text" in msg

    @staticmethod
    def _build_full_text_unavailable_message(pmcid: str) -> str:
        pmcid_label = f" for **{pmcid}**" if pmcid else ""
        return (
            f"⚠️ **Full text unavailable{pmcid_label}**\n\n"
            "The publisher does not expose full-text XML for this article in PubMed Central.\n\n"
            f"View online at this link: https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/\n\n"
            "**Try another question**\n\n"
        )

    @staticmethod
    def _extract_partial_json_string(raw_text: str, key: str) -> str | None:
        key_pos = raw_text.find(f'"{key}"')
        if key_pos == -1:
            return None

        colon_pos = raw_text.find(":", key_pos)
        if colon_pos == -1:
            return None

        start_quote = raw_text.find('"', colon_pos)
        if start_quote == -1:
            return None

        chars: List[str] = []
        escaped = False
        idx = start_quote + 1
        while idx < len(raw_text):
            ch = raw_text[idx]
            if escaped:
                chars.append("\\" + ch)
                escaped = False
                idx += 1
                continue

            if ch == "\\":
                escaped = True
                idx += 1
                continue

            if ch == '"':
                break

            chars.append(ch)
            idx += 1

        return "".join(chars)

    @staticmethod
    def _unescape_preview(text: str) -> str:
        return (
            text.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\/", "/")
        )

    def _build_synthesis_preview(self, raw_text: str) -> str:
        fields = [
            ("what_the_research_found", "**What the research found:**"),
            ("why_it_matters", "**Why it matters:**"),
            ("the_science_behind_it", "**The science behind it:**"),
        ]

        blocks: List[str] = []
        for key, header in fields:
            value = self._extract_partial_json_string(raw_text, key)
            if value:
                blocks.append(f"{header}\n\n{self._unescape_preview(value)}")

        return "\n\n".join(blocks)

    def _build_qa_preview(self, raw_text: str) -> str:
        answer = self._extract_partial_json_string(raw_text, "answer")
        if not answer:
            return ""
        return f"**Answer:**\n\n{self._unescape_preview(answer)}"

    async def _stream_synthesis(
        self,
        *,
        user_input: str,
        documents: List[Dict[str, str]],
        include_sources: bool = True,
    ) -> AsyncIterator[str]:
        synthesis_messages = build_synthesis_messages(
            user_input, documents, include_sources=include_sources
        )
        try:
            streamed_parts: List[str] = []
            emitted_preview = ""
            async for chunk in self.llm.astream(synthesis_messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    streamed_parts.append(token)
                    preview = self._build_synthesis_preview("".join(streamed_parts))
                    if preview.startswith(emitted_preview):
                        delta = preview[len(emitted_preview) :]
                    else:
                        delta = preview
                    emitted_preview = preview
                    if delta:
                        yield delta

            content = "".join(streamed_parts)
            parsed = ResearchSynthesis.from_llm(content)
            self.last_validated_response = parsed.to_markdown(
                include_sources=include_sources
            )
        except Exception:
            # Fall back to plain streaming if structured parsing fails.
            self.last_validated_response = None

    async def _stream_qa(
        self,
        *,
        user_input: str,
        documents: List[Dict[str, str]],
        chat_history: Optional[List],
    ) -> AsyncIterator[str]:
        qa_messages = build_qa_messages(user_input, documents)

        # Add chat history for context
        if chat_history:
            qa_messages = [
                qa_messages[0],
                *chat_history,
                qa_messages[1],
            ]

        try:
            streamed_parts: List[str] = []
            emitted_preview = ""
            async for chunk in self.llm.astream(qa_messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    streamed_parts.append(token)
                    preview = self._build_qa_preview("".join(streamed_parts))
                    if preview.startswith(emitted_preview):
                        delta = preview[len(emitted_preview) :]
                    else:
                        delta = preview
                    emitted_preview = preview
                    if delta:
                        yield delta

            content = "".join(streamed_parts)
            parsed = ArticleQAAnswer.from_llm(content)
            self.last_validated_response = parsed.to_markdown()
        except Exception:
            # Fall back to plain streaming if structured parsing fails.
            self.last_validated_response = None

    async def astream(self, user_input: str, chat_history: Optional[List] = None):
        """
        Stream the agent's response asynchronously with tool calling support.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history as list of LangChain Message objects

        Yields:
            Chunks of the response for streaming output
        """
        self.last_validated_response = None

        # Compile tool description for the system prompt
        tool_descriptions = build_tool_descriptions(self.tools)

        # Create messages starting with system message
        messages = [
            SystemMessage(
                content=f"""You are a biomedical research assistant that helps non-experts understand medical research. 
                            
                You can help users find and understand scientific literature from PubMed Central.
                {tool_descriptions}

                When users ask about topics or want to find articles, you should use the search_pubmed_central tool.
                When users want to understand the details of a specific article, you should use the retrieve_full_text tool.

                Be conversational, helpful, and informative. Users may not be familiar with scientific terminology, so explain things in simple terms with a low lexile score."""
            )
        ]

        # Add chat history if it exists
        if chat_history:
            messages.extend(chat_history)

        # Add current user message
        messages.append(HumanMessage(content=user_input))

        # Get response from LLM
        response = await self.llm_with_tools.ainvoke(messages)

        # Check if the response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Handle tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                if tool_name in self.tools:
                    normalized_args = self._extract_tool_args(tool_name, tool_args)

                    if tool_name == "search_pubmed_central":
                        query = normalized_args.get("query", "")
                        yield f"🔎 Searching PubMed Central for research on **{query}**...\n\n"
                    elif tool_name == "retrieve_full_text":
                        pmcid = normalized_args.get("pmcid", "")
                        yield f"📄 Retrieving full text for **{pmcid}**...\n\n"

                    try:
                        tool_result = self._run_tool(tool_name, normalized_args)

                        if tool_result:
                            if tool_name == "search_pubmed_central":
                                yield f"📚 Found {len(tool_result)} articles. Let me filter and explain what the research shows...\n\n"
                            elif tool_name == "retrieve_full_text":
                                yield f"📚 Retrieved {len(tool_result)} full-text sections. Let me explain what they show...\n\n"

                            async for chunk in self._stream_synthesis(
                                user_input=user_input,
                                documents=tool_result,
                                include_sources=(tool_name != "retrieve_full_text"),
                            ):
                                yield chunk

                            # Offer to answer follow up question for accessibility
                            yield f"\n\n---\n\n💡 *Any other follow-up questions? Just ask!*"
                        else:
                            yield "No articles found for that query."
                    except Exception as e:
                        # Account for when full text is not available or rights prohibited
                        if (
                            tool_name == "retrieve_full_text"
                            and self._is_full_text_unavailable_error(e)
                        ):
                            pmcid = normalized_args.get("pmcid", "")
                            yield self._build_full_text_unavailable_message(pmcid)
                        else:
                            yield "❌ Something went wrong: Please try again later.\n\n"
        else:
            # No tool calls, check if we have documents for already
            if self.documents:
                async for chunk in self._stream_qa(
                    user_input=user_input,
                    documents=self.documents,
                    chat_history=chat_history,
                ):
                    yield chunk
            else:
                # Just stream the regular response
                if hasattr(response, "content"):
                    yield response.content
                else:
                    yield str(response)

    async def ainvoke(self, user_input: str, chat_history: List = None):
        """
        Invoke the agent and get the complete response.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history

        Returns:
            The complete response
        """
        result = ""
        async for chunk in self.astream(user_input, chat_history):
            result += chunk
        return result
