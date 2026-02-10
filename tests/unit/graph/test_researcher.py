
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.graph.nodes import NodeName
from src.graph.researcher import Researcher
from src.models.states import ResearcherState, Source, SourceType
from src.prompt_templates import MAX_ITERATION_REACHED
from src.tools import ToolManager, ToolName


class TestResearcher:
    """Tests for Researcher node."""

    def test_node_name(self, mock_llm_client):
        """Test that node_name returns RESEARCHER."""
        researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
        assert researcher.node_name == NodeName.RESEARCHER

    def test_initialization_with_tools(self, mock_llm_client):
        """Test that researcher initializes with specified tools."""
        tool_names = [ToolName.WEB_SEARCH]
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool") as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            researcher = Researcher(mock_llm_client, tool_names)

            # Verify tools were retrieved
            mock_get_tool.assert_called_once_with(ToolName.WEB_SEARCH)

            # Verify internal tools list
            assert researcher._tools == [mock_tool]

    def test_extract_max_iterations_valid(self):
        """Test _extract_max_iterations with valid config."""
        researcher = Researcher(Mock(), [])
        config = RunnableConfig(configurable={"max_iterations": 5})
        result = researcher._extract_max_iterations(config)
        assert result == 5

    def test_extract_max_iterations_missing_configurable(self):
        """Test _extract_max_iterations raises ValueError when configurable missing."""
        researcher = Researcher(Mock(), [])
        config = RunnableConfig()
        with pytest.raises(ValueError, match="configurables not passed"):
            researcher._extract_max_iterations(config)

    def test_extract_max_iterations_missing_max_iterations(self):
        """Test _extract_max_iterations raises TypeError when max_iterations missing."""
        researcher = Researcher(Mock(), [])
        config = RunnableConfig(configurable={})
        with pytest.raises(TypeError, match="max_iterations must be int"):
            researcher._extract_max_iterations(config)

    def test_extract_max_iterations_wrong_type(self):
        """Test _extract_max_iterations raises TypeError when max_iterations not int."""
        researcher = Researcher(Mock(), [])
        config = RunnableConfig(configurable={"max_iterations": "five"})
        with pytest.raises(TypeError, match="max_iterations must be int"):
            researcher._extract_max_iterations(config)

    def test_get_system_prompt(self):
        """Test _get_system_prompt returns formatted string."""
        result = Researcher._get_system_prompt()
        assert isinstance(result, str)
        assert "UTC" in result or "Z" in result  # Should contain timestamp

    def test_build_messages(self, sample_researcher_state):
        """Test _build_messages includes system prompt and researcher history."""
        researcher = Researcher(Mock(), [])
        messages = researcher._build_messages(sample_researcher_state)
        assert len(messages) == 1  # Only system message when history empty
        assert isinstance(messages[0], SystemMessage)

        # Add history and test
        history = [HumanMessage(content="Previous")]
        state_with_history = {**sample_researcher_state, "researcher_history": history}
        messages = researcher._build_messages(state_with_history)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[1] == history[0]

    def test_handle_initial_request_with_tool_call(self):
        """Test _handle_initial_request when LLM returns tool call."""
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "test"}, "id": "tool1"}
            ],
        )
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            state: ResearcherState = {
                "planned_subtasks": ["subtask1"],
                "current_iteration": 0,
                "should_continue": False,
                "researcher_history": [],
                "sub_agent_call_id": "test-call-id",
            }
            messages = [SystemMessage(content="System")]

            result = researcher._handle_initial_request(state, messages)

            # Verify client called correctly
            mock_llm_client.with_structured_output.assert_called_once()
            call_args = mock_llm_client.with_structured_output.call_args
            # Should be system message + new HumanMessage with subtasks as string
            assert len(call_args.kwargs["messages"]) == 2
            assert isinstance(call_args.kwargs["messages"][0], SystemMessage)
            assert isinstance(call_args.kwargs["messages"][1], HumanMessage)
            assert call_args.kwargs["messages"][1].content == "['subtask1']"
            assert call_args.kwargs["tools"] == [mock_tool]

            # Verify state updates
            assert result["current_iteration"] == 1
            assert result["should_continue"] is True
            assert len(result["researcher_history"]) == 2
            assert isinstance(result["researcher_history"][0], HumanMessage)
            assert result["researcher_history"][0].content == "['subtask1']"
            assert result["researcher_history"][1] == mock_response

    def test_handle_initial_request_without_tool_call_raises(self):
        """Test _handle_initial_request raises ValueError when no tool call."""
        mock_response = AIMessage(content="No tool call")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            state: ResearcherState = {
                "planned_subtasks": ["subtask"],
                "current_iteration": 0,
                "should_continue": False,
                "researcher_history": [],
                "sub_agent_call_id": "test-call-id",
            }
            messages = [
                SystemMessage(content="System"),
                HumanMessage(content="subtask"),
            ]

            with pytest.raises(
                ValueError, match="Expected Researcher Node to perform tool calling"
            ):
                researcher._handle_initial_request(state, messages)

    def test_handle_subsequent_iterations_with_tool_call(self):
        """Test _handle_subsequent_iterations when LLM returns tool call."""
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "test"}, "id": "tool1"}
            ],
        )
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            # Simulate existing research findings
            researcher.research_findings = [
                Source(source="url1", content="content1", type=SourceType.WEB)
            ]
            state: ResearcherState = {
                "current_iteration": 1,
                "should_continue": True,
                "researcher_history": [
                    HumanMessage(content="subtask"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "web_search",
                                "args": {"query": "test"},
                                "id": "tool1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content='[{"source": "url2", "content": "content2", "type": "web"}]',
                        tool_call_id="tool1",
                    ),
                ],
                "planned_subtasks": [],
                "sub_agent_call_id": "test-call-id",
            }
            messages = [SystemMessage(content="System")]

            result = researcher._handle_subsequent_iterations(state, messages)

            # Should call continue_research
            assert result["current_iteration"] == 2
            assert result["should_continue"] is True
            assert len(result["researcher_history"]) == 1  # Only the new response
            # Research findings should be extended
            assert len(researcher.research_findings) == 2
            assert researcher.research_findings[1]["source"] == "url2"

    def test_handle_subsequent_iterations_without_tool_call(self):
        """Test _handle_subsequent_iterations when LLM returns final response."""
        mock_response = AIMessage(content="Final answer")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            researcher.research_findings = [
                Source(source="url1", content="content1", type=SourceType.WEB)
            ]
            state: ResearcherState = {
                "current_iteration": 1,
                "should_continue": True,
                "researcher_history": [],
                "planned_subtasks": [],
                "sub_agent_call_id": "test-call-id",
            }
            messages = [SystemMessage(content="System")]

            result = researcher._handle_subsequent_iterations(state, messages)

            # Should call finalize_research
            assert result["should_continue"] is False
            assert "message_history" in result
            assert len(result["message_history"]) == 1
            assert isinstance(result["message_history"][0], ToolMessage)
            assert result["message_history"][0].tool_call_id == "test-call-id"
            assert result["message_history"][0].content == "Final answer"
            # Researcher history should contain the response
            assert result["researcher_history"] == [mock_response]

    def test_handle_max_iterations(self):
        """Test _handle_max_iterations when max iterations reached."""
        researcher = Researcher(Mock(), [])
        researcher.research_findings = [
            Source(source="url1", content="content1", type=SourceType.WEB),
            Source(source="url2", content="content2", type=SourceType.WEB),
        ]
        state: ResearcherState = {
            "current_iteration": 10,
            "should_continue": True,
            "researcher_history": [],
            "planned_subtasks": [],
            "sub_agent_call_id": "test-call-id",
        }

        result = researcher._handle_max_iterations(state)

        assert result["should_continue"] is False
        assert "message_history" in result
        assert len(result["message_history"]) == 1
        assert isinstance(result["message_history"][0], ToolMessage)
        assert result["message_history"][0].tool_call_id == "test-call-id"
        # Content should be formatted research synthesis
        assert "Total Sources Gathered: 2" in result["message_history"][0].content
        # Researcher history should contain max iteration message
        assert len(result["researcher_history"]) == 1
        assert result["researcher_history"][0].content == MAX_ITERATION_REACHED

    def test_get_sub_agent_call_id_valid(self):
        """Test _get_sub_agent_call_id returns id when present."""
        researcher = Researcher(Mock(), [])
        state: ResearcherState = {
            "sub_agent_call_id": "test-id",
            "should_continue": False,
            "current_iteration": 0,
            "planned_subtasks": [],
            "researcher_history": [],
        }
        result = researcher._get_sub_agent_call_id(state)
        assert result == "test-id"

    def test_get_sub_agent_call_id_missing(self):
        """Test _get_sub_agent_call_id raises ValueError when missing."""
        researcher = Researcher(Mock(), [])
        state: ResearcherState = {
            "should_continue": False,
            "current_iteration": 0,
            "planned_subtasks": [],
            "researcher_history": [],
            # Missing sub_agent_call_id
        }
        with pytest.raises(ValueError, match="sub_agent_call_id not found"):
            researcher._get_sub_agent_call_id(state)

    def test_execute_initial_request(self):
        """Test _execute routes to _handle_initial_request when current_iteration=0."""
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "test"}, "id": "tool1"}
            ],
        )
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            state: ResearcherState = {
                "planned_subtasks": ["subtask"],
                "current_iteration": 0,
                "should_continue": False,
                "researcher_history": [],
                "sub_agent_call_id": "test-call-id",
            }
            config = RunnableConfig(configurable={"max_iterations": 5})

            result = researcher(state, config)

            assert result["current_iteration"] == 1
            assert result["should_continue"] is True

    def test_execute_max_iterations_reached(self):
        """Test _execute routes to _handle_max_iterations when current_iteration > max."""
        researcher = Researcher(Mock(), [])
        researcher.research_findings = [
            Source(source="url", content="content", type=SourceType.WEB)
        ]
        state: ResearcherState = {
            "current_iteration": 6,
            "should_continue": True,
            "planned_subtasks": [],
            "researcher_history": [],
            "sub_agent_call_id": "test-call-id",
        }
        config = RunnableConfig(configurable={"max_iterations": 5})

        result = researcher(state, config)

        assert result["should_continue"] is False
        assert "message_history" in result

    def test_execute_subsequent_iterations(self):
        """Test _execute routes to _handle_subsequent_iterations for middle iterations."""
        mock_response = AIMessage(content="Final answer")
        mock_llm_client = Mock()
        mock_llm_client.with_structured_output.return_value = mock_response
        mock_tool = Mock()
        with patch.object(ToolManager, "get_structured_tool", return_value=mock_tool):
            researcher = Researcher(mock_llm_client, [ToolName.WEB_SEARCH])
            state: ResearcherState = {
                "current_iteration": 2,
                "should_continue": True,
                "planned_subtasks": [],
                "researcher_history": [],
                "sub_agent_call_id": "test-call-id",
            }
            config = RunnableConfig(configurable={"max_iterations": 5})

            result = researcher(state, config)

            assert result["should_continue"] is False
            assert "message_history" in result
