
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.graph.nodes import NodeName
from src.graph.orchestrator import OrchestratorNode
from src.models.states import OrchestratorState
from src.tools import ToolManager, ToolName


class TestOrchestratorNode:
    """Tests for OrchestratorNode."""

    def test_node_name(self, mock_llm_client):
        """Test that node_name returns ORCHESTRATOR."""
        orchestrator = OrchestratorNode(mock_llm_client)
        assert orchestrator.node_name == NodeName.ORCHESTRATOR

    def test_preprocess_system_prompt(self):
        """Test _preprocess_system_prompt returns formatted string."""
        result = OrchestratorNode._preprocess_system_prompt()
        assert isinstance(result, str)
        assert "UTC" in result or "Z" in result  # Should contain timestamp

    def test_execute_with_tool_call_delegation(self):
        """Test execution when LLM returns a tool call (should delegate)."""
        # Mock LLM client response with tool call
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": ToolName.RESEARCH_TOOL,
                    "args": {"subtasks": ["subtask1", "subtask2"]},
                    "id": "test-research-id",
                }
            ],
        )
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response

        # Mock ToolManager
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            orchestrator = OrchestratorNode(mock_llm_client)
            state: OrchestratorState = {
                "message_history": [],
                "should_delegate": False,
                "planned_subtasks": [],
                "sub_agent_call_id": "",
                "response": "",
            }
            config = RunnableConfig()

            result = orchestrator(state, config)

            # Verify client was called with correct arguments
            mock_llm_client.with_structured_output.assert_called_once()
            call_args = mock_llm_client.with_structured_output.call_args
            # Should be called with keyword arguments
            assert "messages" in call_args.kwargs
            assert "tools" in call_args.kwargs
            assert (
                len(call_args.kwargs["messages"]) == 1
            )  # only system message when history empty
            from langchain_core.messages import SystemMessage

            assert isinstance(call_args.kwargs["messages"][0], SystemMessage)
            assert call_args.kwargs["tools"] == [mock_tool]

            # Verify state updates
            assert result["should_delegate"] is True
            assert result["planned_subtasks"] == ["subtask1", "subtask2"]
            assert result["sub_agent_call_id"] == "test-research-id"
            assert result["message_history"] == [mock_response]
            assert "response" not in result or result.get("response") == ""

    def test_execute_without_tool_call_final_response(self):
        """Test execution when LLM returns final response (no delegation)."""
        mock_response = AIMessage(content="Final answer")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response

        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            orchestrator = OrchestratorNode(mock_llm_client)
            state: OrchestratorState = {
                "message_history": [],
                "should_delegate": False,
                "planned_subtasks": [],
                "sub_agent_call_id": "",
                "response": "",
            }
            config = RunnableConfig()

            result = orchestrator(state, config)

            assert result["should_delegate"] is False
            assert result["response"] == "Final answer"
            assert result["message_history"] == [mock_response]
            assert (
                "planned_subtasks" not in result or result.get("planned_subtasks") == []
            )
            assert (
                "sub_agent_call_id" not in result
                or result.get("sub_agent_call_id") == ""
            )

    def test_execute_with_existing_message_history(self):
        """Test that existing message history is included in LLM call."""
        mock_response = AIMessage(content="Response")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response

        existing_message = AIMessage(content="Previous message")
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            orchestrator = OrchestratorNode(mock_llm_client)
            state: OrchestratorState = {
                "message_history": [existing_message],
                "should_delegate": False,
                "planned_subtasks": [],
                "sub_agent_call_id": "",
                "response": "",
            }
            config = RunnableConfig()

            result = orchestrator(state, config)

            # Verify client was called with system message + existing history
            mock_llm_client.with_structured_output.assert_called_once()
            call_args = mock_llm_client.with_structured_output.call_args
            # Should be called with keyword arguments
            assert "messages" in call_args.kwargs
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2  # system + existing message
            from langchain_core.messages import SystemMessage

            assert isinstance(messages[0], SystemMessage)
            assert messages[1] == existing_message

            # New response should be added to history
            assert result["message_history"] == [mock_response]

    def test_execute_tool_call_missing_research_id(self):
        """Test that missing research id in tool call raises exception."""
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": ToolName.RESEARCH_TOOL,
                    "args": {"subtasks": []},
                    "id": "",  # Empty string to trigger missing research id error
                }
            ],
        )
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response

        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            orchestrator = OrchestratorNode(mock_llm_client)
            state: OrchestratorState = {
                "message_history": [],
                "should_delegate": False,
                "planned_subtasks": [],
                "sub_agent_call_id": "",
                "response": "",
            }
            config = RunnableConfig()

            with pytest.raises(Exception, match="Research id not provided"):
                orchestrator(state, config)

    def test_execute_config_ignored(self):
        """Test that config parameter is ignored (underscore convention)."""
        mock_response = AIMessage(content="Response")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response

        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            orchestrator = OrchestratorNode(mock_llm_client)
            state: OrchestratorState = {
                "message_history": [],
                "should_delegate": False,
                "planned_subtasks": [],
                "sub_agent_call_id": "",
                "response": "",
            }
            config = RunnableConfig(callbacks=[], tags=["test"])

            # Should not raise any errors
            result = orchestrator(state, config)
            assert result["message_history"] == [mock_response]
