from datetime import UTC, datetime

from client import OpenAIClient
from graph.base import BaseNode
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from models.states import OrchestratorState
from prompt_templates import SYS_ORCHESTRATOR
from tools import ShouldResearch, ToolManager

from .exceptions import NodeInputError, NodeOutputError, StateError
from .nodes import NodeName


class OrchestratorNode(BaseNode):
    """Orchestrator node that decides whether to perform research
    and synthesizes results."""

    NODE_NAME = NodeName.ORCHESTRATOR

    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client

    @property
    def node_name(self) -> NodeName:
        return NodeName.ORCHESTRATOR

    @staticmethod
    def _preprocess_system_prompt() -> str:
        """Format the system prompt with the current UTC time."""
        return SYS_ORCHESTRATOR.format(current_time=datetime.now(UTC).isoformat())

    @staticmethod
    def _research_factory() -> StructuredTool:
        """Return a ready-to-use research tool."""
        return ToolManager.get_structured_tool("research_tool")

    @staticmethod
    def _is_tool_call(response: AIMessage) -> bool:
        """Check if the AIMessage contains a tool call."""
        return bool(response.tool_calls)

    def _execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the orchestrator logic.

        Case 1: Initial decision - determine if research is needed
        Case 2: Post-research - synthesize findings into response

        Args:
            state: Current orchestrator state

        Returns:
            Updated orchestrator state
        """
        system_message = SystemMessage(content=self._preprocess_system_prompt())
        message_history = state.get("message_history", [])
        messages = [system_message, *message_history]

        # that mean we just initiated the graph
        if not state.get("should_research", False):
            return self._handle_initial_decision(state, messages)
        # that means we invoked research and now its time for synthesis
        return self._handle_research_synthesis(state, messages)

    def _handle_initial_decision(
        self, state: OrchestratorState, messages: list[BaseMessage]
    ) -> OrchestratorState:
        """Handle initial decision on whether to research.

        Args:
            state: Current orchestrator state
            messages: Message history with system prompt

        Returns:
            Updated state with research decision

        Raises:
            NodeInputError: If research_id is missing in tool call
            NodeOutputError: If neither tool calls nor text response produced by the LLM
        """
        response = self.client.with_structured_output(
            messages=messages,
            tools=[self._research_factory()],
        )

        if self._is_tool_call(response):
            tool_call = response.tool_calls[0]
            research_id = tool_call.get("id")
            if not research_id:
                raise NodeInputError(
                    node_name=self.node_name,
                    message="Research id not provided in tool call",
                )

            planned_subtasks = ShouldResearch(**tool_call["args"])
            return OrchestratorState(
                message_history=[*state.get("message_history", []), response],
                should_research=True,
                planned_subtasks=planned_subtasks.subtasks,
                research_id=research_id,
            )

        text_response = response.content
        if not text_response:
            raise NodeOutputError(
                node_name=self.node_name,
                message="Neither Tool calls nor text response produced by the LLM",
            )

        return OrchestratorState(
            message_history=[*state.get("message_history", []), response],
            should_research=False,
            response=text_response,
        )

    def _handle_research_synthesis(
        self, state: OrchestratorState, messages: list[BaseMessage]
    ) -> OrchestratorState:
        """Synthesize research findings into final response.

        Args:
            state: Current orchestrator state with research findings
            messages: Message history with system prompt

        Returns:
            Updated state with synthesized response

        Raises:
            StateError: If research findings or ID are missing
        """

        updated_messages = [*messages]
        response = self.client.chat(messages=updated_messages)

        response_content = response.content
        if not response_content:
            raise NodeOutputError(
                node_name=self.node_name,
                message="Empty response content from LLM after research",
            )

        return OrchestratorState(
            message_history=[*state.get("message_history", []), response],
            response=response_content,
            should_research=False,
        )
