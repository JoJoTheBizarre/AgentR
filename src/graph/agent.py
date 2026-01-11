from client import OpenAIClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from models.states import AgentState
from tools.web_search_tool import web_search_factory

from .nodes import NodeName
from .orchestrator import OrchestratorNode
from .process_query import QueryProcessor
from .researcher import Researcher


class AgentR:
    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client
        self.graph = self._build_graph()

    def _build_initial_state(self, request: str) -> AgentState:
        """Create initial state for the agent graph."""
        return AgentState(
            query=request,
            response="",
            message_history=[],
            should_research=False,
            should_continue=False,
            current_iteration=0,
            max_iteration=5,
            planned_subtasks=[],
            research_id="",
            research_findings=[],
            researcher_history=[],
        )

    def invoke(self, request: str):
        initial_state = self._build_initial_state(request)
        agent_response = self.graph.invoke(initial_state)
        string_response = agent_response.get("response")
        return string_response

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node(NodeName.PREPROCESSOR, QueryProcessor())
        graph_builder.add_node(NodeName.RESEARCHER, Researcher(self.client))
        graph_builder.add_node(NodeName.ORCHESTRATOR, OrchestratorNode(self.client))
        graph_builder.add_node(
            NodeName.TOOL_NODE,
            ToolNode(tools=[web_search_factory()], messages_key="researcher_history"),
        )

        graph_builder.add_edge(START, NodeName.PREPROCESSOR)
        graph_builder.add_edge(NodeName.PREPROCESSOR, NodeName.ORCHESTRATOR)
        graph_builder.add_conditional_edges(
            NodeName.ORCHESTRATOR,
            self._should_continue,
            [NodeName.RESEARCHER, END]
        )

        graph_builder.add_conditional_edges(
            NodeName.RESEARCHER,
            self._should_continue_research,
            [NodeName.TOOL_NODE, NodeName.ORCHESTRATOR]
        )


        return graph_builder.compile()


    @staticmethod
    def _should_continue(state: AgentState):
        if state.get("research_id"):
            return NodeName.RESEARCHER
        else:
            return END

    @staticmethod
    def _should_continue_research(state: AgentState):
        should_continue = state.get("should_continue")
        if should_continue:
            return NodeName.TOOL_NODE
        else:
            return NodeName.ORCHESTRATOR
