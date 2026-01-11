from datetime import UTC, datetime

from client import OpenAIClient
from graphs.base import BaseNode
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from models.states import ResearcherState
from prompt_templates import RESEARCH_PROMPT
from tools import web_search_factory


class Researcher(BaseNode):
    """Process user queries and add them to message history."""

    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client

    @staticmethod
    def _preprocess_system_prompt() -> str:
        """Format the system prompt with the current UTC time."""
        return RESEARCH_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    @staticmethod
    def _handle_initial_request(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        subtasks = state.get("planned_subtasks")
        request = HumanMessage(content=str(subtasks))
        messages = [*messages, request]
        response = self.client.with_structured_output(
            messages=messages, tools=[web_search_factory], parallel=True
        )
        return ResearcherState(
            current_iteration=state.get("current_iteration") + 1,
            researcher_history=[state.get("researcher_history"), request, response],
        )

    @staticmethod
    def _should_continue(response: AIMessage) -> bool:
        if response.tool_calls:
            return True
        else:
            return False

    @staticmethod
    def _handle_subsequent_iterations(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        response = self.client.with_structured_output(
            messages=messages, tools=[web_search_factory], parallel=True
        )
        should_continue = self._should_continue(response)
        if should_continue:
            return ResearcherState(
                current_iteration=state.get("current_iteration") + 1,
                researcher_history=[state.get("researcher_history"), response],
                should_continue=True
            )
        return ResearcherState(
            message_history=state.get("message_history")
            + [ToolMessage(content=response, tool_call_id=state.get("research_id"))],
            should_continue=False
        )

    def _execute(self, state: ResearcherState) -> ResearcherState:
        current_iteration = state.get("current_iteration")
        system_message = SystemMessage(content=self._preprocess_system_prompt())
        messages = [system_message, *state.get("researcher_history")]
        if current_iteration == 0:
            return self._handle_initial_request(state, messages)

        else:
            return self._handle_subsequent_iterations(state, messages)
