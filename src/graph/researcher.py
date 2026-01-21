import json
from datetime import UTC, datetime

from client import OpenAIClient
from exceptions import AgentExecutionError, ValidationError
from graph.base import BaseNode
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from models.states import ResearcherState, Source, SourceType
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

    def _parse_research_results(self, results_str: str) -> list[dict]:
        """Parse and validate research results JSON.

        Args:
            results_str: JSON string containing research results

        Returns:
            List of validated source dictionaries

        Raises:
            ValidationError: If JSON is invalid or structure doesn't match Source schema
        """
        try:
            parsed = json.loads(results_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in research results: {e}")

        if not isinstance(parsed, list):
            raise ValidationError(f"Expected list of sources, got {type(parsed).__name__}")

        validated_sources = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValidationError(f"Source at index {i} is not a dictionary")

            # Check required fields
            required_fields = {"source", "content", "type"}
            missing = required_fields - set(item.keys())
            if missing:
                raise ValidationError(f"Source at index {i} missing fields: {missing}")

            # Validate type is valid SourceType
            try:
                SourceType(item["type"])
            except ValueError:
                raise ValidationError(
                    f"Source at index {i} has invalid type: {item['type']}. "
                    f"Expected one of: {[e.value for e in SourceType]}"
                )

            # Ensure fields are strings
            if not isinstance(item["source"], str):
                raise ValidationError(f"Source at index {i} 'source' field must be string")
            if not isinstance(item["content"], str):
                raise ValidationError(f"Source at index {i} 'content' field must be string")

            validated_sources.append(item)

        return validated_sources

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
            parsed_results = self._parse_research_results(research_results)
            self.research_findings.extend([Source(**item) for item in parsed_results])

            return ResearcherState(
                current_iteration=1,
                researcher_history=[response],
                should_continue=True,
            )

        sub_agent_call_id = state.get("sub_agent_call_id", "")

        return ResearcherState(
            message_history=[
                ToolMessage(content=str(response.content), tool_call_id=sub_agent_call_id)
            ],
            researcher_history=[response],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
        )

    def _handle_research_handoff(
        self, state: ResearcherState, messages: list[BaseMessage]
    ) -> ResearcherState:
        iteration_limit_message = AIMessage(content=MAX_ITERATION_REACHED)

        synthesis = self._format_research_synthesis(self.research_findings)

        sub_agent_call_id = state.get("sub_agent_call_id", "")

        return ResearcherState(
            message_history=[
                ToolMessage(
                    content=synthesis,
                    tool_call_id=sub_agent_call_id,
                )
            ],
            researcher_history=[iteration_limit_message],
            should_continue=False,
            planned_subtasks=[],
            sub_agent_call_id="",
        )

    def _execute(self, state: ResearcherState, config: RunnableConfig) -> ResearcherState:
        configurables = config.get("configurable")
        if not configurables:
            raise AgentExecutionError("could not find max_iteration config")
        max_iterations = configurables.get("max_iterations")
        current_iteration = state.get("current_iteration", 0)
        system_message = SystemMessage(content=self._preprocess_system_prompt())
        researcher_history = state.get("researcher_history", [])
        messages = [system_message, *researcher_history]

        if current_iteration == 0:
            return self._handle_initial_request(state, messages)
        elif current_iteration > max_iterations:
            return self._handle_research_handoff(state, messages)
        else:
            return self._handle_subsequent_iterations(state, messages)
