from graphs.base import BaseNode
from langchain_core.messages import HumanMessage
from models.states import PreprocessorState
from client import OpenAIClient
from models.states import ResearcherState


class Researcher(BaseNode):
    """Process user queries and add them to message history."""

    def __init__(self, llm_client: OpenAIClient) -> None:
        self.client = llm_client

    def _execute(self, state: ResearcherState) -> ResearcherState:
        """Add the user query to message history as a HumanMessage.

        Args:
            state: Current preprocessor state containing the query

        Returns:
            Updated state with query added to message history

        Raises:
            ValueError: If query is missing from state
        """
        query = state.get("query")
        if not query:
            raise ValueError("Could not find user query in state")

        return state