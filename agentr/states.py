from enum import StrEnum
from operator import add
from typing import Annotated
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, NotRequired


class SourceType(StrEnum):
    """Types of information sources."""

    WEB = "web"
    DOCUMENT = "document"


class Source(TypedDict):
    """Information source structure."""

    source: str
    content: str
    type: SourceType


class AgentState(TypedDict):
    """Main agent state."""

    query: str
    response: str
    message_history: Annotated[list[BaseMessage], add]
    should_delegate: bool
    should_continue: bool
    current_iteration: int
    planned_subtasks: list[str]
    sub_agent_call_id: str
    researcher_history: Annotated[list[BaseMessage], add]


class PreprocessorState(TypedDict):
    """Preprocessor node state."""

    query: NotRequired[str]
    message_history: list[BaseMessage]


class OrchestratorState(TypedDict):
    """Orchestrator node state."""

    message_history: list[BaseMessage]
    should_delegate: bool
    planned_subtasks: NotRequired[list[str]]
    sub_agent_call_id: NotRequired[str]
    response: NotRequired[str]


class ResearcherState(TypedDict):
    """Researcher node state."""

    current_iteration: NotRequired[int]
    planned_subtasks: NotRequired[list[str]]
    researcher_history: list[BaseMessage]
    should_continue: bool
    sub_agent_call_id: NotRequired[str]
    message_history: NotRequired[list[BaseMessage]]
