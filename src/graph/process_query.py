from graph.base import BaseNode
from langchain_core.messages import HumanMessage
from models.states import PreprocessorState

from .exceptions import StateError
from .nodes import NodeName


class QueryProcessor(BaseNode):
    """Process user queries and add them to message history."""

    NODE_NAME = NodeName.PREPROCESSOR

    def __init__(self) -> None:
        pass

    @property
    def node_name(self) -> NodeName:
        return NodeName.PREPROCESSOR

    def _execute(self, state: PreprocessorState) -> PreprocessorState:
        """Add the user query to message history as a HumanMessage.

        Args:
            state: Current preprocessor state containing the query

        Returns:
            Updated state with query added to message history

        Raises:
            StateError: If query is missing from state
        """
        query = state.get("query")
        if not query:
            raise StateError(
                message="Could not find user query in state", state_field="query"
            )

        return PreprocessorState(message_history=[HumanMessage(content=query)])
