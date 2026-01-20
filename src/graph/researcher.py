import json
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
from models.states import ResearcherState, Source, Sources
from prompt_templates import (
    MAX_ITERATION_REACHED,
    RESEARCH_PROMPT,
    RESEARCH_SYNTHESIS_TEMPLATE,
)
from tools import ToolManager, ToolName

from .nodes import NodeName


class Researcher(BaseNode):
    """Process user queries and add them to message history."""

    NODE_NAME = NodeName.RESEARCHER

    def __init__(
        self, llm_client: OpenAIClient, tool_names: list[ToolName | str] | None = None
    ) -> None:
        self.client = llm_client
        self.tool_names = tool_names or [ToolName.WEB_SEARCH]
        self._tools = [
            ToolManager.get_structured_tool(name) for name in self.tool_names
        ]

        self.research_findings: list[Source] = []

    @property
    def node_name(self) -> NodeName:
        return NodeName.RESEARCHER

    @staticmethod
    def _preprocess_system_prompt() -> str:
        """Format the system prompt with the current UTC time."""
        return RESEARCH_PROMPT.format(current_time=datetime.now(UTC).isoformat())

    @staticmethod
    def _should_continue(response: AIMessage) -> bool:
        return bool(
            response.tool_calls
        )

    @staticmethod
    def _format_research_synthesis(research_findings: list[Source]) -> str:
        """Format research findings using the synthesis template."""

        def format_single_source(idx: int, source: Source) -> str:
            return f"""[Source {idx + 1}]
            Type: {source['type']}
            Source: {source['source']}
            Content: {source['content']}
            """

        formatted_sources = "\n".join(
            format_single_source(i, source)
            for i, source in enumerate(research_findings)
        )

        return RESEARCH_SYNTHESIS_TEMPLATE.format(
            total_sources=len(research_findings), formatted_sources=formatted_sources
        )

    def _handle_initial_request(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        #clear the research findings from previous runs if they exist
        self.research_findings = []
        subtasks = state.get("planned_subtasks", [])
        request = HumanMessage(content=str(subtasks))
        messages = [*messages, request]

        response = self.client.with_structured_output(
            messages=messages,
            tools=self._tools,
            parallel=True,
        )

        return ResearcherState(
            current_iteration=1,
            researcher_history=[request, response],
            should_continue=self._should_continue(response),
        )

    def _handle_subsequent_iterations(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:

        response = self.client.with_structured_output(
            messages=messages,
            tools=self._tools,
            parallel=True,
        )

        should_continue = self._should_continue(response)

        if should_continue:
            research_results = str(state["researcher_history"][-1].content)
            parsed_results = json.loads(research_results)
            self.research_findings.extend([Source(**item) for item in parsed_results])

            return ResearcherState(
                current_iteration=1,
                researcher_history=[response],
                should_continue=True,
            )

        research_id = state.get("research_id", "")

        return ResearcherState(
            message_history=[
                ToolMessage(content=str(response.content), tool_call_id=research_id)
            ],
            researcher_history=[response],
            should_continue=False,
            planned_subtasks=[],
            research_id="",
        )

    def _handle_research_handoff(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        iteration_limit_message = AIMessage(content=MAX_ITERATION_REACHED)

        synthesis = self._format_research_synthesis(self.research_findings)

        research_id = state.get("research_id", "")

        return ResearcherState(
            message_history=[
                ToolMessage(
                    content=synthesis,
                    tool_call_id=research_id,
                )
            ],
            researcher_history=[iteration_limit_message],
            should_continue=False,
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
        elif current_iteration > 5:
            return self._handle_research_handoff(state, messages)
        else:
            return self._handle_subsequent_iterations(state, messages)
