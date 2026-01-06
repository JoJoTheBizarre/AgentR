from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from src.agentr.agent.nodes import ResearchNode
from src.agentr.core import NodeType, ResearchAgentState

class ResearchAgent:
    def __init__(self, *args, **kwargs):
        self._compiled = False
        self.graph = None
        pass

    def _compile(self):
        if self._compiled:
            return self.graph
        else:
            raise Exception("graph not compiled")
        pass

    def _assemble_graph(self):
        graph = StateGraph(start_node=START, end_node=END, state_type=ResearchAgentState)

        graph.add_node(NodeType.RESEARCH, ResearchNode())
        graph.add_node(NodeType.TOOL, ToolNode(tool_name="search_tool"))

        graph.add_edge(START, "research_node_1")
        graph.add_edge("research_node_1", END)

        self.graph = graph
        self._compiled = True
        return graph