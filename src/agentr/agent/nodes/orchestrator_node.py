from langchain_core.runnables import RunnableConfig

from agentr.agent.nodes.base import BaseNode
from agentr.client import LLMClient
from agentr.core.state import AgentState
from agentr.tools import ToolRegistry


class OrchestratorNode(BaseNode[AgentState, AgentState]):
    def __init__(
        self, name: str, llm_client: LLMClient, tool_registry: ToolRegistry
    ) -> None:
        super().__init__(name)
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    async def _execute(
        self, state: AgentState, config: RunnableConfig | None = None
    ) -> AgentState:
        """Orchestrates by planning steps and decomposing the query."""
        return state
