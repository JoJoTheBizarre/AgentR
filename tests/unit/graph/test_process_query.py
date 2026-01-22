"""
Tests for QueryProcessor node.
"""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.graph.process_query import QueryProcessor
from src.models.states import PreprocessorState


class TestQueryProcessor:
    """Tests for QueryProcessor node."""

    def test_node_name(self):
        """Test that node_name returns PREPROCESSOR."""
        processor = QueryProcessor()
        assert processor.node_name.value == "preprocessor"

    def test_execute_with_query(self):
        """Test processing state with a query."""
        processor = QueryProcessor()
        state: PreprocessorState = {
            "query": "Test research question",
            "message_history": [],
        }
        config = RunnableConfig()

        result = processor(state, config)

        assert "message_history" in result
        assert len(result["message_history"]) == 1
        assert isinstance(result["message_history"][0], HumanMessage)
        assert result["message_history"][0].content == "Test research question"

    def test_execute_with_existing_message_history(self):
        """Test that existing message history is replaced (not appended)."""
        processor = QueryProcessor()
        # PreprocessorState doesn't preserve existing message_history
        # According to implementation, it returns new list with only the HumanMessage
        state: PreprocessorState = {
            "query": "New query",
            "message_history": [HumanMessage(content="Old message")],
        }
        config = RunnableConfig()

        result = processor(state, config)

        # Should have only one message (the new query)
        assert len(result["message_history"]) == 1
        assert result["message_history"][0].content == "New query"

    def test_execute_without_query_raises_exception(self):
        """Test that missing query raises an exception."""
        processor = QueryProcessor()
        state: PreprocessorState = {"message_history": []}  # Missing query
        config = RunnableConfig()

        with pytest.raises(Exception, match="Could not find user query in state"):
            processor(state, config)

    def test_execute_with_empty_query_string(self):
        """Test processing with empty query string raises exception."""
        processor = QueryProcessor()
        state: PreprocessorState = {"query": "", "message_history": []}
        config = RunnableConfig()

        with pytest.raises(Exception, match="Could not find user query in state"):
            processor(state, config)

    def test_execute_ignores_config(self):
        """Test that config parameter is ignored (underscore convention)."""
        processor = QueryProcessor()
        state: PreprocessorState = {"query": "Test", "message_history": []}
        config = RunnableConfig(callbacks=[], tags=["test"])

        # Should not raise any errors
        result = processor(state, config)

        assert "message_history" in result
        assert len(result["message_history"]) == 1

    def test_state_type_compatibility(self):
        """Test that QueryProcessor works with PreprocessorState type."""
        processor = QueryProcessor()
        # This is a type check - ensure the class is properly typed
        state: PreprocessorState = {"query": "Typed test", "message_history": []}
        config = RunnableConfig()

        result = processor(state, config)
        # Should return PreprocessorState
        assert "message_history" in result
        assert isinstance(result["message_history"], list)
        assert all(isinstance(msg, HumanMessage) for msg in result["message_history"])
