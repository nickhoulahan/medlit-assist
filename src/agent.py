from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class OllamaAgent:

    def __init__(
        self, model: str, tools: List = None, temperature: float = 0.3
    ):
        """
        Ollama agent with tools.

        Args:
            model: The Ollama model to use
            tools: List of LangChain tools to provide to the agent
            temperature: Model temperature (0-1)
        """
        self.model = model
        self.tools_list = tools or []
        self.tools = {tool.name: tool for tool in self.tools_list}
        self.documents = []  # for storing fetched documents

        # LangChain ChatOllama instance
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

        # Try to bind tools to the model if supported
        if self.tools_list:
            try:
                self.llm_with_tools = self.llm.bind_tools(self.tools_list)
            except Exception:
                self.llm_with_tools = self.llm
        else:
            self.llm_with_tools = self.llm

    def _tool_descriptions(self) -> str:
        if not self.tools:
            return ""
        return "\n\nAvailable tools:\n" + "\n".join(
            [f"- {name}: {tool.description}" for name, tool in self.tools.items()]
        )

    @staticmethod
    def _format_documents_context(documents: List[Dict[str, str]]) -> str:
        return "\n\n".join(
            [
                f"Article {i+1} (PMC ID: {doc['pmcid']}):\n{doc['citation']}\n\nAbstract: {doc['abstract']}"
                for i, doc in enumerate(documents)
            ]
        )

    def _extract_tool_args(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        query = tool_args.get("query", "")
        max_results = tool_args.get("max_results", 3)
        return {"query": query, "max_results": max_results}

    def _run_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Dict[str, str]]:
        """Execute a single tool call and normalize storage for follow-up Q&A."""
        tool = self.tools[tool_name]
        result = tool.invoke(tool_args)

        # Store documents in memory for follow-up Q&A.
        # (Keeping as list[dict] to preserve existing app/tests expectations.)
        self.documents = result
        return result

    async def _stream_synthesis(self, *, user_input: str, documents: List[Dict[str, str]]) -> AsyncIterator[str]:
        context = self._format_documents_context(documents)
        synthesis_messages = [
            SystemMessage(
                content="""You are a biomedical research communicator. Your job is to:

1. Read scientific research articles
2. Explain the key findings in simple, accessible language that anyone can understand with a low lexile score
3. Always cite which article (by number and PMC ID) you're referencing
4. Use analogies and plain language - avoid jargon
5. Be accurate but approachable
6. Structure your response with clear sections

Format your response like:
**What the research found:**

**Why it matters:**

**The science behind it:**

Always cite sources with a link to the article like: (Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678)"""
            ),
            HumanMessage(
                content=f"""Based on these research articles, please explain what we know about: {user_input}

Research Articles:
{context}

Remember: Explain in simple, everyday language while staying accurate. Cite the articles with pubmed central links."""
            ),
        ]

        async for chunk in self.llm.astream(synthesis_messages):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)


    async def astream(self, user_input: str, chat_history: Optional[List] = None):
        """
        Stream the agent's response asynchronously with tool calling support.

        Args:
            user_input: The user's message
            chat_history: Previous conversation history as list of LangChain Message objects

        Yields:
            Chunks of the response for streaming output
        """
        # Compile tool description for the system prompt
        tool_descriptions = self._tool_descriptions()

        # Create messages starting with system message
        messages = [
            SystemMessage(
                content=f"""You are a biomedical research assistant that helps non-experts understand medical research. 
                            
                You can help users find and understand scientific literature from PubMed Central.
                {tool_descriptions}

                When users ask about topics or want to find articles, you should use the search_pubmed_central tool.

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
                    normalized_args = self._extract_tool_args(tool_args)
                    query = normalized_args.get("query", "")

                    yield f"🔎 Searching PubMed Central for research on **{query}**...\n\n"

                    try:
                        tool_result = self._run_tool(tool_name, normalized_args)

                        if tool_result:
                            yield f"📚 Found {len(tool_result)} articles. Let me filter and explain what the research shows...\n\n"

                            async for chunk in self._stream_synthesis(
                                user_input=user_input, documents=tool_result
                            ):
                                yield chunk

                            # Offer to answer follow up question for accessibility
                            yield f"\n\n---\n\n💡 *Any other follow-up questions? Just ask!*"
                        else:
                            yield "No articles found for that query."
                    except Exception as e:
                        yield f"❌ Error: {str(e)}"
        else:
            # No tool calls, check if we have documents for already
            if self.documents:
                # Create context from documents
                context = self._format_documents_context(self.documents)

                # Create Q&A prompt with simple language requirement
                qa_messages = [
                    SystemMessage(
                        content=f"""You are a biomedical research communicator. Answer questions based on the 
                        research articles provided. 

                        Research Articles:
                        {context}

                        Guidelines:
                        - Explain in simple, everyday language (like explaining to a friend)
                        - Always cite which article(s) you're referencing (e.g., "Article 1, PMC12345678")
                        - Avoid medical jargon, or explain it if necessary
                        - Be accurate but accessible with a low lexile score
                        - If the articles don't answer the question, say so clearly

                        Your goal is to make medical research understandable to everyone."""
                    )
                ]

                # Add chat history for context
                if chat_history:
                    qa_messages.extend(chat_history)

                # Add current question
                qa_messages.append(HumanMessage(content=user_input))

                # Stream the Q&A response
                async for chunk in self.llm.astream(qa_messages):
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    else:
                        yield str(chunk)
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
