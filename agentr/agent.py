import logging
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .client import OpenAIClient
from .config import EnvConfig, get_default_runtime_config
from .exceptions import ResponseError
from .nodes import OrchestratorNode, QueryProcessor, Researcher
from .states import AgentState
from .tools import ToolManager, ToolName

logger = logging.getLogger(__name__)


class AgentR:
    """AI Research Assistant Agent with in-memory conversation history."""

    def __init__(
        self,
        llm_client: OpenAIClient,
        env_config: EnvConfig,
        *,
        tracing: bool = False,
        thread_id: str = "default",
        enable_memory: bool = True,
    ):
        """Initialize AgentR with in-memory persistence."""
        self.client = llm_client
        self.researcher_tools = [ToolName.WEB_SEARCH]
        self.callbacks = []
        self.thread_id = thread_id
        self.enable_memory = enable_memory

        # Setup in-memory checkpointer
        self.memory_saver = MemorySaver() if enable_memory else None

        if tracing:
            self._setup_tracing(env_config)

        self.config = RunnableConfig(
            callbacks=self.callbacks,
            configurable={
                **get_default_runtime_config(),
                "thread_id": self.thread_id,
            },
        )

        self.graph = self._build_graph()
        logger.info(
            f"AgentR initialized: thread_id={self.thread_id}, "
            f"memory={'enabled' if enable_memory else 'disabled'}"
        )

    def _setup_tracing(self, env_config: EnvConfig):
        """Setup Langfuse tracing."""
        lf_callback = CallbackHandler()
        lf_callback.client = Langfuse(
            public_key=env_config.langfuse_public_key,
            secret_key=env_config.langfuse_secret_key,
            base_url=env_config.langfuse_base_url,
        )
        self.callbacks.append(lf_callback)
        logger.info("Langfuse tracing enabled")

    def _build_initial_state(self, request: str) -> AgentState:
        """Create initial state for graph execution."""
        return AgentState(
            query=request,
            response="",
            message_history=[],
            should_delegate=False,
            should_continue=False,
            current_iteration=0,
            planned_subtasks=[],
            sub_agent_call_id="",
            researcher_history=[],
        )

    def invoke(self, request: str) -> str:
        """Execute agent on request and return response."""
        initial_state = self._build_initial_state(request)
        agent_response = self.graph.invoke(initial_state, self.config)

        literal_response = agent_response.get("response")
        if literal_response:
            return literal_response
        else:
            raise ResponseError("Agent execution completed but response is empty")

    def stream(self, request: str):
        """Stream agent execution with memory support."""
        initial_state = self._build_initial_state(request)
        for event in self.graph.stream(initial_state, self.config):
            yield event

    def get_state(self) -> dict:
        """Get current conversation state from memory."""
        if not self.enable_memory or not self.memory_saver:
            return {}

        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.memory_saver.get(config)
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return {}

    def get_message_history(self) -> list:
        """Get conversation message history."""
        state = self.get_state()
        return state.get("message_history", [])

    def clear_memory(self):
        """Clear conversation memory for current thread."""
        if not self.enable_memory or not self.memory_saver:
            logger.warning("Memory not enabled")
            return

        try:
            if hasattr(self.memory_saver, "storage"):
                if self.thread_id in self.memory_saver.storage:
                    del self.memory_saver.storage[self.thread_id]
                    logger.info(f"Memory cleared for thread: {self.thread_id}")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the agent graph with optional memory."""
        ToolManager.initialize()

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("preprocessor", QueryProcessor())
        graph_builder.add_node("orchestrator", OrchestratorNode(self.client))
        graph_builder.add_node(
            "researcher", Researcher(self.client, tool_names=self.researcher_tools)
        )
        graph_builder.add_node(
            "tool_node",
            ToolNode(
                tools=[ToolManager.get_tool(name) for name in self.researcher_tools],
                messages_key="researcher_history",
            ),
        )

        graph_builder.add_edge(START, "preprocessor")
        graph_builder.add_edge("preprocessor", "orchestrator")
        graph_builder.add_edge("tool_node", "researcher")

        graph_builder.add_conditional_edges(
            "orchestrator", self._should_continue, ["researcher", END]
        )

        graph_builder.add_conditional_edges(
            "researcher", self._should_continue_research, ["tool_node", "orchestrator"]
        )

        if self.enable_memory and self.memory_saver:
            compiled_graph = graph_builder.compile(checkpointer=self.memory_saver)
            logger.info("Graph compiled with in-memory checkpointing")
        else:
            compiled_graph = graph_builder.compile()
            logger.info("Graph compiled without memory")

        return compiled_graph

    @staticmethod
    def _should_continue(state: AgentState):
        """Determine if research delegation is needed."""
        return "researcher" if state.get("should_delegate") else END

    @staticmethod
    def _should_continue_research(state: AgentState):
        """Determine if more research iterations are needed."""
        return "tool_node" if state.get("should_continue") else "orchestrator"
