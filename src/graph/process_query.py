from graph.base import BaseNode
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from models.states import PreprocessorState

from .nodes import NodeName


class QueryProcessor(BaseNode):
    """Process user queries and add them to message history."""

    NODE_NAME = NodeName.PREPROCESSOR

    def __init__(self) -> None:
        pass

    @property
    def node_name(self) -> NodeName:
        return NodeName.PREPROCESSOR

    def _execute(
        self, state: PreprocessorState, config: RunnableConfig
    ) -> PreprocessorState:
        """Add the user query to message history as a HumanMessage."""
        _ = config
        query = state.get("query")
        if not query:
            raise Exception("Could not find user query in state")

        return PreprocessorState(message_history=[HumanMessage(content=query)])
