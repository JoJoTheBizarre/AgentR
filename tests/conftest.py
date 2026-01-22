"""
Shared pytest fixtures and configuration for AgentR tests.
"""

import asyncio
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.config import EnvConfig
from src.models.states import AgentState, ResearcherState
from src.tools.manager import ToolManager


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Mock environment variables for testing."""
    env_vars = {
        "API_KEY": "test-api-key",
        "API_URL": "https://api.test.com",
        "MODEL_NAME": "test-model",
        "TAVILY_API_KEY": "test-tavily-key",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "development",
        "LANGFUSE_PUBLIC_KEY": "test-public-key",
        "LANGFUSE_SECRET_KEY": "test-secret-key",
        "LANGFUSE_BASE_URL": "http://localhost:3000",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield


@pytest.fixture
def env_config(mock_env_vars) -> EnvConfig:
    """Create an EnvConfig instance with mocked environment."""
    return EnvConfig()  # type: ignore


@pytest.fixture
def mock_openai_client() -> Generator[Mock, None, None]:
    """Mock OpenAI client with controlled responses."""
    with patch("src.client.openai_client.ChatOpenAI") as mock_chat_openai:
        instance = Mock()
        instance.bind_tools = Mock(return_value=instance)
        instance.with_structured_output = Mock(return_value=instance)
        instance.invoke = Mock()
        mock_chat_openai.return_value = instance
        yield instance


@pytest.fixture
def mock_tavily_client() -> Generator[AsyncMock, None, None]:
    """Mock Tavily API client."""
    with patch("src.tools.execution.web_search_tool.TavilyClient") as mock_tavily:
        instance = AsyncMock()
        instance.search = AsyncMock()
        mock_tavily.return_value = instance
        yield instance


@pytest.fixture
def mock_langfuse() -> Generator[Mock, None, None]:
    """Mock Langfuse for tracing tests."""
    with patch("src.graph.agent.Langfuse") as mock_langfuse_class:
        instance = Mock()
        mock_langfuse_class.return_value = instance
        yield instance


@pytest.fixture
def clean_tool_registry() -> Generator[None, None, None]:
    """Clean tool registry before/after each test."""
    ToolManager.clear_registry()
    ToolManager._initialized = False
    yield
    ToolManager.clear_registry()
    ToolManager._initialized = False


@pytest.fixture
def sample_agent_state() -> AgentState:
    """Create a sample AgentState for testing."""
    return {
        "query": "Test research query",
        "response": "",
        "message_history": [],
        "should_delegate": False,
        "should_continue": False,
        "current_iteration": 0,
        "planned_subtasks": [],
        "sub_agent_call_id": "",
        "researcher_history": [],
    }


@pytest.fixture
def sample_researcher_state() -> ResearcherState:
    """Create a sample ResearcherState for testing."""
    return {
        "should_continue": False,
        "current_iteration": 0,
        "planned_subtasks": ["Research subtask 1", "Research subtask 2"],
        "researcher_history": [],
        "sub_agent_call_id": "test-call-id",
    }


@pytest.fixture
def sample_ai_message() -> AIMessage:
    """Create a sample AIMessage for testing."""
    return AIMessage(content="Test AI response")


@pytest.fixture
def sample_human_message() -> HumanMessage:
    """Create a sample HumanMessage for testing."""
    return HumanMessage(content="Test human query")


@pytest.fixture
def mock_openai_response() -> dict:
    """Create a mock OpenAI API response."""
    return {
        "content": "Test response from OpenAI",
        "tool_calls": [],
    }


@pytest.fixture
def mock_tool_call_response() -> dict:
    """Create a mock OpenAI response with tool calls."""
    return {
        "content": "",
        "tool_calls": [{"name": "web_search", "args": {"query": "test search query"}}],
    }


@pytest.fixture
def mock_tavily_response() -> dict:
    """Create a mock Tavily search response."""
    return {
        "query": "test search",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "Test content 1",
                "score": 0.9,
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "Test content 2",
                "score": 0.8,
            },
        ],
    }


@pytest.fixture
def mock_llm_client() -> Generator[Mock, None, None]:
    """Mock OpenAIClient wrapper for graph nodes."""
    with patch("src.client.OpenAIClient") as mock_openai_wrapper:
        instance = Mock()
        instance.with_structured_output = Mock()
        instance.chat = Mock()
        mock_openai_wrapper.return_value = instance
        yield instance


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
