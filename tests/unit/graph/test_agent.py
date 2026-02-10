from unittest.mock import Mock, patch

import pytest
from langgraph.graph import END, START

from src.exceptions import ResponseError
from src.graph.agent import AgentR
from src.graph.nodes import NodeName
from src.models.states import AgentState
from src.tools import ToolManager, ToolName


class TestAgentR:
    """Tests for AgentR class."""

    def test_initialization_without_tracing(self, mock_llm_client, env_config):
        """Test AgentR initialization without tracing."""
        agent = AgentR(mock_llm_client, env_config, tracing=False)
        assert agent.client == mock_llm_client
        assert agent.researcher_tools == [ToolName.WEB_SEARCH]
        assert agent.callbacks == []
        assert agent.config is not None
        assert agent.graph is not None

    def test_initialization_with_tracing(self, mock_llm_client, env_config):
        """Test AgentR initialization with tracing enabled."""
        with (
            patch("src.graph.agent.Langfuse") as mock_langfuse,
            patch("src.graph.agent.CallbackHandler") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            agent = AgentR(mock_llm_client, env_config, tracing=True)

            # Verify tracing setup
            mock_langfuse.assert_called_once_with(
                public_key=env_config.langfuse_public_key,
                secret_key=env_config.langfuse_secret_key,
                base_url=env_config.langfuse_base_url,
            )
            assert len(agent.callbacks) == 1
            assert agent.callbacks[0] == mock_handler_instance

    def test_build_initial_state(self, mock_llm_client, env_config):
        """Test _build_initial_state creates proper AgentState."""
        agent = AgentR(mock_llm_client, env_config, tracing=False)
        request = "Test research question"
        state = agent._build_initial_state(request)

        assert state["query"] == request
        assert state["response"] == ""
        assert state["message_history"] == []
        assert state["should_delegate"] is False
        assert state["should_continue"] is False
        assert state["current_iteration"] == 0
        assert state["planned_subtasks"] == []
        assert state["sub_agent_call_id"] == ""
        assert state["researcher_history"] == []

    def test_should_continue_delegates_to_researcher(self):
        """Test _should_continue returns RESEARCHER when should_delegate is True."""
        state: AgentState = {"should_delegate": True}
        result = AgentR._should_continue(state)
        assert result == NodeName.RESEARCHER

    def test_should_continue_ends_when_no_delegation(self):
        """Test _should_continue returns END when should_delegate is False."""
        state: AgentState = {"should_delegate": False}
        result = AgentR._should_continue(state)
        assert result == END

    def test_should_continue_research_continues_to_tool_node(self):
        """Test _should_continue_research returns TOOL_NODE when should_continue is True."""
        state: AgentState = {"should_continue": True}
        result = AgentR._should_continue_research(state)
        assert result == NodeName.TOOL_NODE

    def test_should_continue_research_returns_to_orchestrator(self):
        """Test _should_continue_research returns ORCHESTRATOR when should_continue is False."""
        state: AgentState = {"should_continue": False}
        result = AgentR._should_continue_research(state)
        assert result == NodeName.ORCHESTRATOR

    def test_invoke_successful_with_response(self, mock_llm_client, env_config):
        """Test invoke returns response when graph execution succeeds."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {"response": "Test answer"}
        with patch.object(AgentR, "_build_graph", return_value=mock_graph):
            agent = AgentR(mock_llm_client, env_config, tracing=False)
            result = agent.invoke("Test question")

            assert result == "Test answer"
            mock_graph.invoke.assert_called_once()
            call_args = mock_graph.invoke.call_args
            # Should be called with initial state and config
            assert call_args[0][0]["query"] == "Test question"
            assert (
                call_args[0][1] == agent.config
            )  # config is second positional argument

    def test_invoke_raises_response_error_when_empty(self, mock_llm_client, env_config):
        """Test invoke raises ResponseError when response field is empty."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {"response": ""}
        with patch.object(AgentR, "_build_graph", return_value=mock_graph):
            agent = AgentR(mock_llm_client, env_config, tracing=False)
            with pytest.raises(
                ResponseError,
                match="Agent execution completed but response field is empty",
            ):
                agent.invoke("Test question")

    def test_invoke_raises_response_error_when_missing(
        self, mock_llm_client, env_config
    ):
        """Test invoke raises ResponseError when response field missing."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {}  # No response key
        with patch.object(AgentR, "_build_graph", return_value=mock_graph):
            agent = AgentR(mock_llm_client, env_config, tracing=False)
            with pytest.raises(
                ResponseError,
                match="Agent execution completed but response field is empty",
            ):
                agent.invoke("Test question")

    def test_build_graph_creates_nodes_and_edges(self, mock_llm_client, env_config):
        """Test _build_graph constructs graph with proper nodes and edges."""
        # Mock all external dependencies
        with (
            patch("src.graph.agent.StateGraph") as mock_state_graph,
            patch("src.graph.agent.ToolNode") as mock_tool_node,
            patch.object(ToolManager, "get_structured_tool") as mock_get_tool,
        ):
            mock_graph_builder = Mock()
            mock_state_graph.return_value = mock_graph_builder
            mock_tool_node_instance = Mock()
            mock_tool_node.return_value = mock_tool_node_instance
            mock_tool = Mock()
            mock_get_tool.return_value = mock_tool

            # Create agent - this will call _build_graph() once during __init__
            agent = AgentR(mock_llm_client, env_config, tracing=False)

            # Verify StateGraph was created with AgentState (called once)
            mock_state_graph.assert_called_once_with(AgentState)

            # Verify nodes were added (4 nodes)
            assert mock_graph_builder.add_node.call_count == 4
            # Check each node addition
            calls = mock_graph_builder.add_node.call_args_list
            node_names = [call[0][0] for call in calls]
            assert NodeName.PREPROCESSOR in node_names
            assert NodeName.RESEARCHER in node_names
            assert NodeName.ORCHESTRATOR in node_names
            assert NodeName.TOOL_NODE in node_names

            # Verify edges
            mock_graph_builder.add_edge.assert_any_call(START, NodeName.PREPROCESSOR)
            mock_graph_builder.add_edge.assert_any_call(
                NodeName.PREPROCESSOR, NodeName.ORCHESTRATOR
            )
            mock_graph_builder.add_edge.assert_any_call(
                NodeName.TOOL_NODE, NodeName.RESEARCHER
            )

            # Verify conditional edges
            assert mock_graph_builder.add_conditional_edges.call_count == 2
            # First conditional edge: orchestrator -> researcher/end
            cond1_args = mock_graph_builder.add_conditional_edges.call_args_list[0]
            assert cond1_args[0][0] == NodeName.ORCHESTRATOR
            assert cond1_args[0][1] == agent._should_continue
            assert cond1_args[0][2] == [NodeName.RESEARCHER, END]
            # Second conditional edge: researcher -> tool_node/orchestrator
            cond2_args = mock_graph_builder.add_conditional_edges.call_args_list[1]
            assert cond2_args[0][0] == NodeName.RESEARCHER
            assert cond2_args[0][1] == agent._should_continue_research
            assert cond2_args[0][2] == [NodeName.TOOL_NODE, NodeName.ORCHESTRATOR]

            # Verify graph compilation
            mock_graph_builder.compile.assert_called_once()
            # The agent's graph should be the compiled graph
            assert agent.graph == mock_graph_builder.compile.return_value

    def test_researcher_tools_configurable(self, mock_llm_client, env_config):
        """Test that researcher_tools defaults to WEB_SEARCH only (subclass attribute ignored)."""

        class CustomAgent(AgentR):
            researcher_tools = [ToolName.WEB_SEARCH, ToolName.RESEARCH_TOOL]

        agent = CustomAgent(mock_llm_client, env_config, tracing=False)
        # Note: AgentR.__init__ sets self.researcher_tools = [ToolName.WEB_SEARCH]
        # overriding any subclass attribute
        assert agent.researcher_tools == [ToolName.WEB_SEARCH]
