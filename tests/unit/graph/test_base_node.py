
import pytest
from langchain_core.runnables import RunnableConfig

from src.exceptions import NodeExecutionError
from src.graph.base import BaseNode, BaseState
from src.graph.nodes import NodeName


class TestBaseNode:
    """Tests for BaseNode abstract class."""

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete classes must implement abstract methods."""

        class IncompleteNode(BaseNode):
            # Missing node_name and _execute
            pass

        with pytest.raises(TypeError):
            IncompleteNode()

    def test_concrete_node_implementation(self):
        """Test a complete concrete node implementation."""

        class ConcreteState(BaseState):
            value: int

        class ConcreteNode(BaseNode):
            @property
            def node_name(self) -> NodeName:
                return NodeName.PREPROCESSOR

            def _execute(
                self, state: ConcreteState, config: RunnableConfig
            ) -> ConcreteState:
                return {
                    "message_history": state.get("message_history", []),
                    "value": state.get("value", 0) + 1,
                }

        node = ConcreteNode()
        assert node.node_name == NodeName.PREPROCESSOR

        state: ConcreteState = {"message_history": [], "value": 5}
        config = RunnableConfig()
        result = node(state, config)
        assert result["value"] == 6

    def test_node_execution_error_wrapping(self):
        """Test that exceptions in _execute are wrapped in NodeExecutionError."""

        class ErrorNode(BaseNode):
            @property
            def node_name(self) -> NodeName:
                return NodeName.RESEARCHER

            def _execute(self, state: BaseState, config: RunnableConfig) -> BaseState:
                raise ValueError("Test error")

        node = ErrorNode()
        state: BaseState = {"message_history": []}
        config = RunnableConfig()

        with pytest.raises(NodeExecutionError) as exc_info:
            node(state, config)

        assert exc_info.value.node_name == NodeName.RESEARCHER
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Test error"

    def test_node_name_property(self):
        """Test that node_name property returns NodeName enum."""

        class TestNode(BaseNode):
            @property
            def node_name(self) -> NodeName:
                return NodeName.ORCHESTRATOR

            def _execute(self, state: BaseState, config: RunnableConfig) -> BaseState:
                return state

        node = TestNode()
        assert node.node_name == NodeName.ORCHESTRATOR
        assert isinstance(node.node_name, NodeName)

    def test_config_passed_to_execute(self):
        """Test that config is passed to _execute method."""
        mock_config = RunnableConfig(callbacks=[], tags=["test"])

        class ConfigNode(BaseNode):
            @property
            def node_name(self) -> NodeName:
                return NodeName.PREPROCESSOR

            def _execute(self, state: BaseState, config: RunnableConfig) -> BaseState:
                # Store config in state for verification
                return {"message_history": [], "config_tags": config["tags"]}

        node = ConfigNode()
        state: BaseState = {"message_history": []}
        result = node(state, mock_config)

        assert "config_tags" in result
        assert result["config_tags"] == ["test"]

    def test_state_type_variables(self):
        """Test that BaseNode supports different state types via generics."""

        class StateA(BaseState):
            field_a: str

        class StateB(BaseState):
            field_b: int

        class TypedNode(BaseNode[StateA, StateB]):
            @property
            def node_name(self) -> NodeName:
                return NodeName.PREPROCESSOR

            def _execute(self, state: StateA, config: RunnableConfig) -> StateB:
                return {
                    "message_history": state.get("message_history", []),
                    "field_b": len(state.get("field_a", "")),
                }

        node = TypedNode()
        state_a: StateA = {"message_history": [], "field_a": "hello"}
        config = RunnableConfig()
        result = node(state_a, config)

        assert "field_b" in result
        assert result["field_b"] == 5
        assert "message_history" in result
