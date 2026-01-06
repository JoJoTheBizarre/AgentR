from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agentr.agent.nodes import OrchestratorNode, ResearchNode, SynthesizeNode
from src.agentr.client import LLMClient
from src.agentr.core.state import AgentState, NodeType
from src.agentr.tools import ToolRegistry, get_default_registry


class ResearchAgentBuilder:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        config: RunnableConfig | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry or get_default_registry()
        self.config = config
        self.graph = None

    @staticmethod
    def should_research(state: AgentState) -> NodeType | str:
        if state.get("should_research", False):
            return NodeType.RESEARCH
        return END

    @staticmethod
    def should_synthesize(state: AgentState) -> NodeType | str:
        if state.get("should_synthesize", False):
            return NodeType.SYNTHESIZE
        return END

    @staticmethod
    def should_continue_research(state: AgentState) -> NodeType | str:
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 0)

        if current_iteration < max_iterations:
            return NodeType.RESEARCH
        return NodeType.ORCHESTRATOR

    def build(self) -> StateGraph:
        graph = StateGraph(state_schema=AgentState)

        graph.add_node(
            NodeType.ORCHESTRATOR,
            OrchestratorNode(
                name=NodeType.ORCHESTRATOR,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
            ),
        )

        graph.add_node(
            NodeType.RESEARCH,
            ResearchNode(
                name=NodeType.RESEARCH,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
            ),
        )

        graph.add_node(
            NodeType.SYNTHESIZE,
            SynthesizeNode(
                name=NodeType.SYNTHESIZE,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
            ),
        )

        structured_tools = self.tool_registry.get_all_tools()
        graph.add_node(NodeType.TOOL, ToolNode(tools=structured_tools))

        graph.add_edge(START, NodeType.ORCHESTRATOR)

        graph.add_conditional_edges(
            NodeType.ORCHESTRATOR,
            ResearchAgentBuilder.should_research,
        )

        graph.add_conditional_edges(
            NodeType.RESEARCH,
            ResearchAgentBuilder.should_continue_research,
        )

        graph.add_conditional_edges(
            NodeType.ORCHESTRATOR,
            ResearchAgentBuilder.should_synthesize,
        )

        self.graph = graph
        return graph
