from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as tool_decorator

from src.client.openai_client import OpenAIClient
from src.config.configs import ClientSettings


# Helper function to create a mock StructuredTool
def create_mock_tool(name: str = "mock_tool") -> StructuredTool:
    """Create a mock StructuredTool for testing."""

    @tool_decorator
    def mock_func() -> str:
        return "mock result"

    mock_tool = mock_func
    mock_tool.name = name
    return mock_tool


# Helper to create mock AIMessage
def create_mock_ai_message(
    content: str = "Test response", tool_calls=None
) -> AIMessage:
    """Create a mock AIMessage for testing."""
    return AIMessage(content=content, tool_calls=tool_calls or [])


# Helper to create mock ToolCall
def create_mock_tool_call(
    id: str = "call_123", name: str = "test_tool", args: dict = None
) -> MagicMock:
    """Create a mock ToolCall object."""
    mock_tc = MagicMock()
    mock_tc.id = id
    mock_tc.name = name
    mock_tc.args = args or {"arg1": "value1"}
    return mock_tc


class TestOpenAIClient:
    """Test suite for OpenAIClient class."""

    def test_init_with_client_config(self):
        """Test initialization with ClientSettings."""
        config = ClientSettings(
            api_key="test-api-key",
            api_url="https://api.test.com",
            model_name="gpt-4-test",
        )

        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai:
            client = OpenAIClient(client_config=config)

            # Verify attributes are set from config
            assert client.api_key == "test-api-key"
            assert client.model == "gpt-4-test"
            # api_url is stored but not used in ChatOpenAI initialization
            assert hasattr(client, "api_url")

            # Verify ChatOpenAI was instantiated with correct parameters
            mock_chat_openai.assert_called_once_with(
                model="gpt-4-test", api_key="test-api-key"
            )

    def test_init_with_api_key_and_model(self):
        """Test initialization with direct api_key and model parameters."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai:
            client = OpenAIClient(api_key="direct-key", model="gpt-3.5-turbo-test")

            assert client.api_key == "direct-key"
            assert client.model == "gpt-3.5-turbo-test"

            mock_chat_openai.assert_called_once_with(
                model="gpt-3.5-turbo-test", api_key="direct-key"
            )

    def test_init_raises_value_error_when_insufficient_args(self):
        """Test that ValueError is raised when neither client_config nor both api_key/model are provided."""
        # Missing both api_key and model
        with pytest.raises(ValueError) as exc_info:
            OpenAIClient(api_key=None, model=None)

        assert (
            "You must provide either client_config or both api_key and model."
            in str(exc_info.value)
        )

        # Missing api_key
        with pytest.raises(ValueError) as exc_info:
            OpenAIClient(api_key=None, model="some-model")

        # Missing model
        with pytest.raises(ValueError) as exc_info:
            OpenAIClient(api_key="some-key", model=None)

    def test_chat_returns_ai_message(self):
        """Test that chat method returns AIMessage."""
        # Create client with mocked ChatOpenAI
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Setup mock response
            mock_response = AIMessage(content="Mocked response")
            mock_client_instance.invoke.return_value = mock_response

            # Test with sample messages
            messages = [HumanMessage(content="Hello")]
            response = client.chat(messages)

            # Verify invoke was called with correct arguments
            mock_client_instance.invoke.assert_called_once_with(input=messages)

            # Verify response is correct
            assert response == mock_response
            assert isinstance(response, AIMessage)
            assert response.content == "Mocked response"

    def test_chat_propagates_exception(self):
        """Test that chat method propagates exceptions from the underlying client."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Make invoke raise an exception
            mock_client_instance.invoke.side_effect = Exception("API Error")

            messages = [HumanMessage(content="Hello")]

            with pytest.raises(Exception) as exc_info:
                client.chat(messages)

            assert "API Error" in str(exc_info.value)

    def test_with_structured_output_returns_ai_message_when_no_tool_calls(self):
        """Test with_structured_output returns AIMessage when no tools are called."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools to return a callable that returns AIMessage without tool_calls
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            # Create AIMessage without tool_calls
            mock_response = AIMessage(content="Text response", tool_calls=[])
            mock_bound_llm.invoke.return_value = mock_response

            # Test
            messages = [HumanMessage(content="Hello")]
            tools = [create_mock_tool()]
            response = client.with_structured_output(messages, tools)

            # Verify bind_tools was called correctly
            mock_client_instance.bind_tools.assert_called_once_with(
                tools=tools, parallel=False
            )

            # Verify invoke was called correctly
            mock_bound_llm.invoke.assert_called_once_with(
                input=messages, choice="first"
            )

            # Verify response
            assert response == mock_response
            assert isinstance(response, AIMessage)
            assert response.tool_calls == []

    def test_with_structured_output_returns_tool_call_when_choice_first(self):
        """Test with_structured_output returns single ToolCall when choice='first' and tools are called."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            # Create mock ToolCall objects
            mock_tool_call1 = create_mock_tool_call(id="call_1", name="tool1")
            mock_tool_call2 = create_mock_tool_call(id="call_2", name="tool2")

            # Create AIMessage with tool_calls
            mock_response = MagicMock()
            mock_response.tool_calls = [mock_tool_call1, mock_tool_call2]
            mock_bound_llm.invoke.return_value = mock_response

            # Test
            messages = [HumanMessage(content="Use a tool")]
            tools = [create_mock_tool()]
            response = client.with_structured_output(messages, tools, choice="first")

            # Verify response is the first tool call
            assert response == mock_tool_call1
            assert response.id == "call_1"
            assert response.name == "tool1"

    def test_with_structured_output_returns_list_when_choice_all(self):
        """Test with_structured_output returns list of ToolCalls when choice='all'."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            # Create mock ToolCall objects
            mock_tool_call1 = create_mock_tool_call(id="call_1", name="tool1")
            mock_tool_call2 = create_mock_tool_call(id="call_2", name="tool2")

            # Create AIMessage with tool_calls
            mock_response = MagicMock()
            mock_response.tool_calls = [mock_tool_call1, mock_tool_call2]
            mock_bound_llm.invoke.return_value = mock_response

            # Test with choice='all'
            messages = [HumanMessage(content="Use tools")]
            tools = [create_mock_tool()]
            response = client.with_structured_output(messages, tools, choice="all")

            # Verify response is list of tool calls
            assert isinstance(response, list)
            assert len(response) == 2
            assert response[0] == mock_tool_call1
            assert response[1] == mock_tool_call2

    def test_with_structured_output_with_parallel_true(self):
        """Test with_structured_output passes parallel=True to bind_tools."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            # Create AIMessage without tool_calls for simplicity
            mock_response = AIMessage(content="Response")
            mock_bound_llm.invoke.return_value = mock_response

            # Test with parallel=True
            messages = [HumanMessage(content="Hello")]
            tools = [create_mock_tool()]
            client.with_structured_output(messages, tools, parallel=True)

            # Verify bind_tools was called with parallel=True
            mock_client_instance.bind_tools.assert_called_once_with(
                tools=tools, parallel=True
            )

    def test_with_structured_output_default_parameters(self):
        """Test that with_structured_output uses correct default parameters."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            mock_response = AIMessage(content="Response")
            mock_bound_llm.invoke.return_value = mock_response

            # Call without specifying parallel and choice (should use defaults)
            messages = [HumanMessage(content="Hello")]
            tools = [create_mock_tool()]
            client.with_structured_output(messages, tools)

            # Verify defaults: parallel=False, choice="first"
            mock_client_instance.bind_tools.assert_called_once_with(
                tools=tools, parallel=False
            )
            mock_bound_llm.invoke.assert_called_once_with(
                input=messages, choice="first"
            )

    def test_with_structured_output_propagates_exception(self):
        """Test that with_structured_output propagates exceptions from underlying client."""
        with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai_cls:
            mock_client_instance = MagicMock()
            mock_chat_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="test-key", model="test-model")

            # Mock bind_tools
            mock_bound_llm = MagicMock()
            mock_client_instance.bind_tools.return_value = mock_bound_llm

            # Make invoke raise an exception
            mock_bound_llm.invoke.side_effect = Exception("Tool binding failed")

            messages = [HumanMessage(content="Hello")]
            tools = [create_mock_tool()]

            with pytest.raises(Exception) as exc_info:
                client.with_structured_output(messages, tools)

            assert "Tool binding failed" in str(exc_info.value)
