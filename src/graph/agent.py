from client import OpenAIClient
from config import EnvConfig
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from models.states import AgentState
from tools import ToolManager, ToolName

from .exceptions import ResponseError
from .nodes import NodeName
from .orchestrator import OrchestratorNode
from .process_query import QueryProcessor
from .researcher import Researcher


class AgentR:
    def __init__(
        self,
        llm_client: OpenAIClient,
        env_config: EnvConfig,
        researcher_tools: list[ToolName | str] | None = None,
    ) -> None:
        self.client = llm_client
        self.researcher_tools = researcher_tools or [
            ToolName.WEB_SEARCH
        ]  # supports web_search by default
        lf_callback = CallbackHandler()
        lf_callback.client = Langfuse(
            public_key=env_config.langfuse_public_key,
            secret_key=env_config.langfuse_secret_key,
            base_url=env_config.langfuse_base_url,
        )
        print("initialized yet")
        self.config = RunnableConfig(callbacks=[lf_callback])
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
            planned_subtasks=[],
            research_id="",
            researcher_history=[],
        )

    def invoke(self, request: str) -> str:
        initial_state = self._build_initial_state(request)
        agent_response = self.graph.invoke(initial_state, config=self.config)
        literal_response = agent_response.get("response")
        if literal_response:
            return literal_response
        else:
            raise ResponseError("Agent execution completed but response field is empty")

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node(NodeName.PREPROCESSOR, QueryProcessor())
        graph_builder.add_node(
            NodeName.RESEARCHER,
            Researcher(self.client, tool_names=self.researcher_tools),
        )
        graph_builder.add_node(
            NodeName.ORCHESTRATOR,
            OrchestratorNode(self.client),
        )

        graph_builder.add_node(
            NodeName.TOOL_NODE,
            ToolNode(
                tools=[
                    ToolManager.get_structured_tool(name)
                    for name in self.researcher_tools
                ],
                messages_key="researcher_history",
            ),
        )

        graph_builder.add_edge(START, NodeName.PREPROCESSOR)
        graph_builder.add_edge(NodeName.PREPROCESSOR, NodeName.ORCHESTRATOR)

        graph_builder.add_conditional_edges(
            NodeName.ORCHESTRATOR, self._should_continue, [NodeName.RESEARCHER, END]
        )

        graph_builder.add_conditional_edges(
            NodeName.RESEARCHER,
            self._should_continue_research,
            [NodeName.TOOL_NODE, NodeName.ORCHESTRATOR],
        )

        graph_builder.add_edge(NodeName.TOOL_NODE, NodeName.RESEARCHER)

        return graph_builder.compile().with_config(self.config)

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
