import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage

from agentr import AgentR, OpenAIClient, EnvConfig
from agentr.exceptions import ResponseError
from agentr.tools import ToolManager


class TestAgentRInitialization:
    """Test AgentR initialization."""

    @patch("agentr.agent.ToolManager.initialize")
    def test_init_with_memory_enabled(self, mock_tool_init):
        """Test initialization with memory enabled."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=True, thread_id="test-123")

        assert agent.enable_memory is True
        assert agent.thread_id == "test-123"
        assert agent.memory_saver is not None

    @patch("agentr.agent.ToolManager.initialize")
    def test_init_with_memory_disabled(self, mock_tool_init):
        """Test initialization with memory disabled."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=False)

        assert agent.enable_memory is False
        assert agent.memory_saver is None


class TestAgentRInvoke:
    """Test AgentR invoke method."""

    @patch("agentr.agent.ToolManager.initialize")
    def test_invoke_returns_response(self, mock_tool_init):
        """Test invoke returns agent response."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=False)

        agent.graph = Mock()
        agent.graph.invoke = Mock(return_value={"response": "Test answer"})

        result = agent.invoke("Test question")

        assert result == "Test answer"
        agent.graph.invoke.assert_called_once()

    @patch("agentr.agent.ToolManager.initialize")
    def test_invoke_raises_error_on_empty_response(self, mock_tool_init):
        """Test invoke raises ResponseError on empty response."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=False)

        agent.graph = Mock()
        agent.graph.invoke = Mock(return_value={"response": ""})

        with pytest.raises(ResponseError):
            agent.invoke("Test question")


class TestAgentRMemory:
    """Test AgentR memory features."""

    @patch("agentr.agent.ToolManager.initialize")
    def test_get_state_with_memory_enabled(self, mock_tool_init):
        """Test getting state with memory enabled."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=True)

        mock_checkpoint = Mock()
        mock_checkpoint.values = {"message_history": [HumanMessage(content="test")]}
        agent.memory_saver.get = Mock(return_value=mock_checkpoint)

        state = agent.get_state()

        assert "message_history" in state
        assert len(state["message_history"]) == 1

    @patch("agentr.agent.ToolManager.initialize")
    def test_get_state_with_memory_disabled(self, mock_tool_init):
        """Test getting state with memory disabled."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=False)

        state = agent.get_state()

        assert state == {}

    @patch("agentr.agent.ToolManager.initialize")
    def test_clear_memory(self, mock_tool_init):
        """Test clearing memory."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = Mock(spec=OpenAIClient)
        agent = AgentR(client, config, enable_memory=True, thread_id="test-123")

        agent.memory_saver.storage = {"test-123": "some data"}

        agent.clear_memory()

        assert "test-123" not in agent.memory_saver.storage


class TestOpenAIClient:
    """Test OpenAIClient."""

    @patch("agentr.client.ChatOpenAI")
    def test_init_with_config(self, mock_chat_openai):
        """Test initialization with config."""
        config = EnvConfig(
            api_key="test-key",
            api_url="https://test.com",
            model_name="test-model",
            tavily_api_key="test-tavily",
        )  # type: ignore

        client = OpenAIClient(client_config=config)

        assert client.api_key == "test-key"
        assert client.model == "test-model"


class TestTools:
    """Test tool functionality."""

    def test_tool_manager_initialize(self):
        """Test ToolManager initialization."""
        ToolManager.clear()
        ToolManager.initialize()

        assert ToolManager._initialized is True
        assert len(ToolManager._registry) > 0

    def test_tool_manager_get_tool(self):
        """Test getting tool from manager."""
        ToolManager.clear()
        ToolManager.initialize()

        from agentr.tools import ToolName

        tool = ToolManager.get_tool(ToolName.WEB_SEARCH)

        assert tool is not None
        assert tool.name == "web_search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
