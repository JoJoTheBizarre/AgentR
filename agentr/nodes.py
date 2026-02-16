import logging
from datetime import UTC, datetime
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from .client import OpenAIClient
from .graph_utils import (
    extract_text_response,
    format_research_synthesis,
    is_tool_call,
    parse_research_results,
)
from .prompts import (
    ORCHESTRATOR_PROMPT,
    RESEARCHER_PROMPT,
    MAX_ITERATION_MESSAGE,
)
from .states import (
    OrchestratorState,
    PreprocessorState,
    ResearcherState,
    Source,
    SourceType,
)
from .tools import ShouldResearch, ToolManager, ToolName

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process user queries into message history."""

    def __call__(self, state: PreprocessorState, config: RunnableConfig) -> PreprocessorState:
        """Add user query to message history."""
        query = state.get("query")
        if not query:
            raise ValueError("Query not found in state")

        return PreprocessorState(message_history=[HumanMessage(content=query)])


class OrchestratorNode:
    """Orchestrate between direct response and research delegation."""

    def __init__(self, llm_client: OpenAIClient):
        self.client = llm_client

    @staticmethod
    def _get_system_prompt() -> str:
        """Get formatted system prompt with current time."""
        return ORCHESTRATOR_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    def __call__(self, state: OrchestratorState, config: RunnableConfig) -> OrchestratorState:
        """Execute orchestrator logic."""
        system_message = SystemMessage(content=self._get_system_prompt())
        message_history = state.get("message_history", [])
        messages = [system_message, *message_history]

        response = self.client.with_structured_output(
            messages=messages,
            tools=[ToolManager.get_tool(ToolName.RESEARCH_TOOL)],
        )

        if is_tool_call(response):
            tool_call = response.tool_calls[0]
            research_id = tool_call.get("id")
            if not research_id:
                raise ValueError("Research ID not provided in tool call")

            planned_subtasks = ShouldResearch(**tool_call["args"])
            return OrchestratorState(
                message_history=[response],
                should_delegate=True,
                planned_subtasks=planned_subtasks.subtasks,
                sub_agent_call_id=research_id,
            )

        return OrchestratorState(
            message_history=[response],
            should_delegate=False,
            response=extract_text_response(response),
        )


class Researcher:
    """Conduct iterative research using web search."""

    def __init__(self, llm_client: OpenAIClient, tool_names: list[ToolName]):
        self.client = llm_client
        self.tool_names = tool_names
        self._tools = [ToolManager.get_tool(name) for name in tool_names]
        self.research_findings: list[Source] = []

    @staticmethod
    def _get_system_prompt() -> str:
        """Get formatted system prompt with current time."""
        return RESEARCHER_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    def __call__(self, state: ResearcherState, config: RunnableConfig) -> ResearcherState:
        """Execute researcher logic."""
        configurables = config.get("configurable")
        if not configurables:
            raise ValueError("Configurables not passed in config")

        max_iterations = configurables.get("max_iterations")
        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be int, got {type(max_iterations).__name__}")

        current_iteration = state.get("current_iteration", 0)

        system_message = SystemMessage(content=self._get_system_prompt())
        researcher_history = state.get("researcher_history", [])
        messages = [system_message, *researcher_history]

        if current_iteration == 0:
            return self._handle_initial_request(state, messages)

        if current_iteration > max_iterations:
            return self._handle_max_iterations(state)

        return self._handle_subsequent_iterations(state, messages)

    def _handle_initial_request(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        """Handle first research iteration."""
        subtasks = state.get("planned_subtasks", [])
        request = HumanMessage(content=str(subtasks))
        messages_with_request = [*messages, request]

        response = self.client.with_structured_output(
            messages=messages_with_request,
            tools=self._tools,
        )

        if not is_tool_call(response):
            raise ValueError("Expected tool call from Researcher, got none")

        current_iteration = state.get("current_iteration", 0)
        return ResearcherState(
            current_iteration=current_iteration + 1,
            researcher_history=[request, response],
            should_continue=True,
        )

    def _handle_subsequent_iterations(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        """Handle middle iterations."""
        response = self.client.with_structured_output(
            messages=messages,
            tools=self._tools,
        )

        if is_tool_call(response):
            return self._continue_research(state, response)

        return self._finalize_research(state, response)

    def _continue_research(self, state: ResearcherState, response: AIMessage) -> ResearcherState:
        """Continue research with new findings."""
        last_message_content = str(state["researcher_history"][-1].content)
        parsed_results = parse_research_results(last_message_content)

        new_sources = [Source(**item) for item in parsed_results]
        self.research_findings.extend(new_sources)

        current_iteration = state.get("current_iteration", 0)
        return ResearcherState(
            current_iteration=current_iteration + 1,
            researcher_history=[response],
            should_continue=True,
        )

    def _finalize_research(self, state: ResearcherState, response: AIMessage) -> ResearcherState:
        """Finalize research and return results."""
        sub_agent_call_id = state.get("sub_agent_call_id")
        if not sub_agent_call_id:
            raise ValueError("sub_agent_call_id not found in state")

        return ResearcherState(
            message_history=[
                ToolMessage(content=str(response.content), tool_call_id=sub_agent_call_id)
            ],
            researcher_history=[response],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
            current_iteration=0,
        )

    def _handle_max_iterations(self, state: ResearcherState) -> ResearcherState:
        """Handle max iteration limit."""
        synthesis = format_research_synthesis(self.research_findings)
        sub_agent_call_id = state.get("sub_agent_call_id")
        if not sub_agent_call_id:
            raise ValueError("sub_agent_call_id not found in state")

        return ResearcherState(
            message_history=[ToolMessage(content=synthesis, tool_call_id=sub_agent_call_id)],
            researcher_history=[AIMessage(content=MAX_ITERATION_MESSAGE)],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
            current_iteration=0,
        )
