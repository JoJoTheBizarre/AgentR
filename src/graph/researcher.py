from datetime import UTC, datetime

from client import OpenAIClient
from graph.base import BaseNode
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from models.states import ResearcherState
from prompt_templates import RESEARCH_PROMPT
from tools import ToolManager

from .nodes import NodeName


class Researcher(BaseNode):
    """Process user queries and add them to message history."""

    NODE_NAME = NodeName.RESEARCHER

    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client

    @property
    def node_name(self) -> NodeName:
        return NodeName.RESEARCHER

    @staticmethod
    def _preprocess_system_prompt() -> str:
        """Format the system prompt with the current UTC time."""
        return RESEARCH_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    @staticmethod
    def _should_continue(response: AIMessage) -> bool:
        return bool(response.tool_calls)

    def _handle_initial_request(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        subtasks = state.get("planned_subtasks", [])
        request = HumanMessage(content=str(subtasks))
        messages = [*messages, request]

        response = self.client.with_structured_output(
            messages=messages,
            tools=[ToolManager.get_structured_tool("web_search")],
            parallel=True,
        )

        researcher_history = state.get("researcher_history", [])

        return ResearcherState(
            current_iteration=state.get("current_iteration", 0) + 1,
            researcher_history=[*researcher_history, request, response],
            should_continue=self._should_continue(response),
        )

    def _handle_subsequent_iterations(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        response = self.client.with_structured_output(
            messages=messages,
            tools=[ToolManager.get_structured_tool("web_search")],
            parallel=True,
        )

        should_continue = self._should_continue(response)

        if should_continue:
            return ResearcherState(
                current_iteration=state.get("current_iteration", 0) + 1,
                researcher_history=[response],
                should_continue=True,
            )

        research_id = state.get("research_id", "")
        message_history = [
            *(state.get("message_history") or []),
            ToolMessage(content=str(response.content), tool_call_id=research_id),
        ]
        should_continue = False

        return ResearcherState(
            message_history=message_history,
            researcher_history=[response],
            should_continue=should_continue,
            planned_subtasks=[],
            research_id="",
        )

    def _execute(self, state: ResearcherState) -> ResearcherState:
        current_iteration = state.get("current_iteration", 0)
        system_message = SystemMessage(content=self._preprocess_system_prompt())
        researcher_history = state.get("researcher_history", [])
        messages = [system_message, *researcher_history]

        if current_iteration == 0:
            return self._handle_initial_request(state, messages)
        else:
            return self._handle_subsequent_iterations(state, messages)
