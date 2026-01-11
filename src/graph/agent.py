from client import OpenAIClient
from langgraph.graph import START, CompiledStateGraph, StateGraph, END
from langgraph.prebuilt import ToolNode
from models.states import AgentState
from tools.web_search_tool import web_search_factory

from .orchestrator import OrchestratorNode
from .process_query import QueryProcessor
from .researcher import Researcher


class AgentR:
    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client
        self.graph = self._build_graph()
        pass
    
    def _build_initial_state()-> AgentState:
        pass
    def invoke(request: str):
        initial_state = self._build_initial_state()
        Agent_response = self.graph.invoke(initial_state)
        string_response = agent_response.get("response")
        return string_response

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("preprocessor", QueryProcessor())
        graph_builder.add_node("researcher", Researcher(self.client))
        graph_builder.add_node("orchestrator", OrchestratorNode(self.client))
        graph_builder.add_node(
            "tool_node",
            ToolNode(tools=[web_search_factory()], messages_key="researcher_history"),
        )

        graph_builder.add_edge(START, "preprocessor")
        graph_builder.add_edge("preprocessor", "orchestrator")
        graph_builder.add_conditional_edges(
            "orchestrator",
            self._should_continue,
            ["researcher", "end"]
        )

        graph_builder.add_conditional_edges(
            "researcher",
            self._should_continue,
            ["tool_node", "orchestrator"]
        )

        
        graph_builder.compile()


    @staticmethod
    def _should_continue(state: AgentState):
        if state.get("research_id"):
            return "researcher"
        else:
            return "end"
        
    @staticmethod
    def _should_continue_research(state: AgentState):
        should_continue = state.get("should_continue")
        if should_continue:
            return "tool_node"
        else:
            "orchestrator"
        