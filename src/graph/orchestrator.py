from datetime import UTC, datetime

from client import OpenAIClient
from graph.base import BaseNode
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from models.states import OrchestratorState
from prompt_templates import SYS_ORCHESTRATOR
from tools import ShouldResearch, ToolManager, ToolName

from .exceptions import NodeInputError, NodeOutputError
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
    def _is_tool_call(response: AIMessage) -> bool:
        """Check if the AIMessage contains a tool call."""
        return bool(
            response.tool_calls
        )  # invoking tool calls means we will delegate control to a subagent

    @staticmethod
    def _extract_text_response(response: AIMessage) -> str:
        content = response.content

        if isinstance(content, str):
            return content

        raise TypeError(
            f"Expected response.content to be str, got {type(content).__name__}"
        )

    @staticmethod
    def _just_started(messages: list[BaseMessage]) -> bool:
        return isinstance(messages[-1], HumanMessage)

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
        if self._just_started:
            return self._handle_initial_decision(state, messages)
        # that means we invoked research and now its time for synthesis
        return self._handle_research_synthesis(state, messages)

    def _handle_initial_decision(
        self, state: OrchestratorState, messages: list[BaseMessage]
    ) -> OrchestratorState:
        """Handle initial decision on whether to research."""
        response = self.client.with_structured_output(
            messages=messages,
            tools=[ToolManager.get_structured_tool(ToolName.RESEARCH_TOOL)],
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
                message_history=[response],
                should_research=True,
                planned_subtasks=planned_subtasks.subtasks,
                sub_agent_call_id=research_id,
            )

        response_content = response.content
        if not response_content:
            raise NodeOutputError(
                node_name=self.node_name,
                message="Neither Tool calls nor text response produced by the LLM",
            )

        return OrchestratorState(
            message_history=[*state.get("message_history", []), response],
            should_research=False,
            response=self._extract_text_response(response),
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

        response = self.client.chat(messages=messages)

        response_content = response.content
        if not response_content:
            raise NodeOutputError(
                node_name=self.node_name,
                message="Empty response content from LLM after research",
            )

        return OrchestratorState(
            message_history=[*state.get("message_history", []), response],
            response=self._extract_text_response(response),
            should_research=False,
        )
