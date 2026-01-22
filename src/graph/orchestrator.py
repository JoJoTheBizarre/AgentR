from datetime import UTC, datetime

from client import OpenAIClient
from graph.base import BaseNode
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from models.states import OrchestratorState
from prompt_templates import SYS_ORCHESTRATOR
from tools import ShouldResearch, ToolManager, ToolName

from .nodes import NodeName
from .utils import extract_text_response, is_tool_call


class OrchestratorNode(BaseNode):
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

    def _execute(
        self, state: OrchestratorState, config: RunnableConfig
    ) -> OrchestratorState:
        _ = config
        system_message = SystemMessage(content=self._preprocess_system_prompt())
        message_history = state.get("message_history", [])
        messages = [system_message, *message_history]

        response = self.client.with_structured_output(
            messages=messages,
            tools=[ToolManager.get_structured_tool(ToolName.RESEARCH_TOOL)],
        )
        if is_tool_call(response):
            # assume there is one toolcall for now
            tool_call = response.tool_calls[0]
            research_id = tool_call.get("id")
            if not research_id:
                raise Exception("Research id not provided in tool call")
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
