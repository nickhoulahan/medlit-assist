import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import OllamaAgent


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.temperature = 0.3
    llm.ainvoke = AsyncMock()
    llm.astream = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


class TestOllamaAgent:

    @patch('src.agent.ChatOllama')
    def test_agent_initialization_without_tools(self, mock_ollama):

        mock_ollama.return_value = MagicMock()
        
        agent = OllamaAgent(model="gpt-oss:20b", temperature=0.3)
        
        assert agent.model == "gpt-oss:20b"
        assert agent.tools_list == []
        assert agent.tools == {}
        assert agent.documents == []
        assert agent.llm is not None

    @patch('src.agent.ChatOllama')
    def test_agent_initialization_with_tools(self, mock_ollama):

        mock_ollama.return_value = MagicMock()
        
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        
        agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
        
        assert len(agent.tools_list) == 1
        assert "test_tool" in agent.tools
        assert agent.tools["test_tool"] == mock_tool

    @patch('src.agent.ChatOllama')
    def test_agent_custom_temperature(self, mock_ollama):

        mock_instance = MagicMock()
        mock_instance.temperature = 0.7
        mock_ollama.return_value = mock_instance
        
        agent = OllamaAgent(model="gpt-oss:20b", temperature=0.7)
        mock_ollama.assert_called_with(model="gpt-oss:20b", temperature=0.7)

    @pytest.mark.asyncio
    @patch('src.agent.ChatOllama')
    async def test_astream_without_tool_calls(self, mock_ollama):

        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []
        
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        agent = OllamaAgent(model="gpt-oss:20b")
        agent.llm_with_tools = mock_llm
        
        chunks = []
        async for chunk in agent.astream("test query"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert "Test response" in "".join(chunks)

    @pytest.mark.asyncio
    @patch('src.agent.ChatOllama')
    async def test_astream_with_chat_history(self, mock_ollama):

        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm
        
        chat_history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer")
        ]
        
        mock_response = MagicMock()
        mock_response.content = "New response"
        mock_response.tool_calls = []
        
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        agent = OllamaAgent(model="gpt-oss:20b")
        agent.llm_with_tools = mock_llm
        
        chunks = []
        async for chunk in agent.astream("new question", chat_history):
            chunks.append(chunk)
        
        # Verify ainvoke was called
        assert mock_llm.ainvoke.called
        
        # Verify chat history was included
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert any(isinstance(msg, HumanMessage) and msg.content == "Previous question" for msg in call_args)

    @pytest.mark.asyncio
    @patch('src.agent.ChatOllama')
    async def test_astream_with_tool_calls(self, mock_ollama):

        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm
        
        mock_tool = MagicMock()
        mock_tool.name = "search_pubmed_central"
        mock_tool.description = "Search for articles"
        mock_tool.invoke.return_value = [
            {
                "pmcid": "PMC123456",
                "citation": "Test citation",
                "abstract": "Test abstract"
            }
        ]
        
        agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
        agent.llm_with_tools = mock_llm
        agent.llm = mock_llm
        
        # Mock the response with tool calls
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "name": "search_pubmed_central",
                "args": {"query": "diabetes", "max_results": 3}
            }
        ]
        
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock the synthesis stream
        async def mock_astream_gen(*args, **kwargs):
            mock_chunk = MagicMock()
            mock_chunk.content = "Synthesis content"
            yield mock_chunk
        
        mock_llm.astream = mock_astream_gen
        
        chunks = []
        async for chunk in agent.astream("search for diabetes"):
            chunks.append(chunk)
        
        # Verify tool was called
        mock_tool.invoke.assert_called_once()
        
        # Verify documents were stored
        assert len(agent.documents) == 1
        assert agent.documents[0]["pmcid"] == "PMC123456"
        
        # Verify output contains expected elements
        full_output = "".join(chunks)
        assert "Searching PubMed Central" in full_output

    @pytest.mark.asyncio
    @patch('src.agent.ChatOllama')
    async def test_astream_with_stored_documents_qa(self, mock_ollama):

        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm
        
        agent = OllamaAgent(model="gpt-oss:20b")
        agent.llm_with_tools = mock_llm
        agent.llm = mock_llm
        
        # Pre-populate documents
        agent.documents = [
            {
                "pmcid": "PMC123456",
                "citation": "Test citation",
                "abstract": "Test abstract about diabetes"
            }
        ]
        
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = []
        
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock the Q&A stream
        async def mock_astream_gen(*args, **kwargs):
            mock_chunk = MagicMock()
            mock_chunk.content = "Answer based on articles"
            yield mock_chunk
        
        mock_llm.astream = mock_astream_gen
        
        chunks = []
        async for chunk in agent.astream("What did the research find?"):
            chunks.append(chunk)
        
        full_output = "".join(chunks)
        assert "Answer based on articles" in full_output

    @pytest.mark.asyncio
    @patch('src.agent.ChatOllama')
    async def test_ainvoke_method(self, mock_ollama):

        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "Complete response"
        mock_response.tool_calls = []
        
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        agent = OllamaAgent(model="gpt-oss:20b")
        agent.llm_with_tools = mock_llm
        
        result = await agent.ainvoke("test query")
        
        assert isinstance(result, str)
        assert "Complete response" in result

    @patch('src.agent.ChatOllama')
    def test_tool_binding_failure_fallback(self, mock_ollama):

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(side_effect=Exception("Binding failed"))
        mock_ollama.return_value = mock_llm
        
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        
        agent = OllamaAgent(model="gpt-oss:20b", tools=[mock_tool])
        
        # Should still initialize without error
        assert agent.llm_with_tools is not None

    @patch('src.agent.ChatOllama')
    def test_documents_storage(self, mock_ollama):

        mock_ollama.return_value = MagicMock()
        
        agent = OllamaAgent(model="gpt-oss:20b")
        
        test_docs = [
            {"pmcid": "PMC1", "citation": "Doc 1", "abstract": "Abstract 1"},
            {"pmcid": "PMC2", "citation": "Doc 2", "abstract": "Abstract 2"}
        ]
        
        agent.documents = test_docs
        
        assert len(agent.documents) == 2
        assert agent.documents[0]["pmcid"] == "PMC1"
        assert agent.documents[1]["pmcid"] == "PMC2"

