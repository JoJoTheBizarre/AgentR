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
from langchain_core.runnables import RunnableConfig
from models.states import ResearcherState, Source
from prompt_templates import MAX_ITERATION_REACHED, RESEARCH_PROMPT
from tools import ToolManager, ToolName

from .nodes import NodeName
from .utils import (
    format_research_synthesis,
    is_tool_call,
    parse_research_results,
)


class Researcher(BaseNode):
    """Process user queries and conduct iterative research."""

    NODE_NAME = NodeName.RESEARCHER

    def __init__(
        self, llm_client: OpenAIClient, tool_names: list[ToolName | str]
    ) -> None:
        self.client = llm_client
        self.tool_names = tool_names
        self._tools = [ToolManager.get_structured_tool(name) for name in tool_names]
        self.research_findings: list[Source] = []

    @property
    def node_name(self) -> NodeName:
        return NodeName.RESEARCHER

    def _execute(
        self, state: ResearcherState, config: RunnableConfig
    ) -> ResearcherState:
        """Execute researcher node logic."""
        max_iterations = self._extract_max_iterations(config)
        current_iteration = state.get("current_iteration", 0)
        messages = self._build_messages(state)

        # Route to appropriate handler based on iteration state
        if current_iteration == 0:
            return self._handle_initial_request(state, messages)

        if current_iteration > max_iterations:
            return self._handle_max_iterations(state)

        return self._handle_subsequent_iterations(state, messages)

    def _extract_max_iterations(self, config: RunnableConfig) -> int:
        """Extract and validate max_iterations from config."""
        configurables = config.get("configurable")
        if configurables is None:
            raise ValueError("configurables not passed during Agent invocation")

        max_iterations = configurables.get("max_iterations")
        if not isinstance(max_iterations, int):
            raise TypeError(
                f"max_iterations must be int, got {type(max_iterations).__name__}"
            )

        return max_iterations

    def _build_messages(self, state: ResearcherState) -> list[BaseMessage]:
        """Build message list with system prompt and history."""
        system_message = SystemMessage(content=self._get_system_prompt())
        researcher_history = state.get("researcher_history", [])
        return [system_message, *researcher_history]

    @staticmethod
    def _get_system_prompt() -> str:
        """Format the system prompt with the current UTC time."""
        return RESEARCH_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    def _handle_initial_request(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        """Handle first iteration of research."""
        subtasks = state.get("planned_subtasks", [])
        request = HumanMessage(content=str(subtasks))
        messages_with_request = [*messages, request]

        response = self.client.with_structured_output(
            messages=messages_with_request,
            tools=self._tools,
        )

        if not is_tool_call(response):
            raise ValueError(
                "Expected Researcher Node to perform tool calling, "
                "received no tool calls instead"
            )

        current_iteration = state.get("current_iteration", 0)
        return ResearcherState(
            current_iteration=current_iteration + 1,
            researcher_history=[request, response],
            should_continue=True,
        )

    def _handle_subsequent_iterations(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        """Handle middle iterations of research."""
        response = self.client.with_structured_output(
            messages=messages,
            tools=self._tools,
        )

        if is_tool_call(response):
            return self._continue_research(state, response)

        return self._finalize_research(state, response)

    def _continue_research(
        self, state: ResearcherState, response: AIMessage
    ) -> ResearcherState:
        """Continue research with new findings."""
        #extract and parse latest research results
        last_message_content = str(state["researcher_history"][-1].content)
        parsed_results = parse_research_results(last_message_content)

        #accumulate findings
        new_sources = [Source(**item) for item in parsed_results]
        self.research_findings.extend(new_sources)

        current_iteration = state.get("current_iteration", 0)
        return ResearcherState(
            current_iteration=current_iteration + 1,
            researcher_history=[response],
            should_continue=True,
        )

    def _finalize_research(
        self, state: ResearcherState, response: AIMessage
    ) -> ResearcherState:
        """Finalize research and return results."""
        sub_agent_call_id = self._get_sub_agent_call_id(state)

        return ResearcherState(
            message_history=[
                ToolMessage(
                    content=str(response.content),
                    tool_call_id=sub_agent_call_id
                )
            ],
            researcher_history=[response],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
            current_iteration=0,
        )

    def _handle_max_iterations(self, state: ResearcherState) -> ResearcherState:
        """Handle max iteration limit reached."""
        synthesis = format_research_synthesis(self.research_findings)
        sub_agent_call_id = self._get_sub_agent_call_id(state)

        return ResearcherState(
            message_history=[
                ToolMessage(content=synthesis, tool_call_id=sub_agent_call_id)
            ],
            researcher_history=[AIMessage(content=MAX_ITERATION_REACHED)],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
            current_iteration=0,
        )

    def _get_sub_agent_call_id(self, state: ResearcherState) -> str:
        """Extract and validate sub_agent_call_id from state."""
        sub_agent_call_id = state.get("sub_agent_call_id")
        if not sub_agent_call_id:
            raise ValueError(
                "sub_agent_call_id not found in state, cannot handle handoff"
            )
        return sub_agent_call_id
