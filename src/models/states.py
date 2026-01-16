from enum import StrEnum
from operator import add
from typing import Annotated, NotRequired

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class SourceType(StrEnum):
    WEB = "web"
    DOCUMENT = "document"


class Source(TypedDict):
    source: str
    content: str
    type: SourceType


Sources = list[Source]


class AgentState(TypedDict):
    query: str
    response: str

    message_history: Annotated[list[BaseMessage], add]

    should_research: bool
    should_continue: bool
    current_iteration: Annotated[int, add]

    planned_subtasks: list[str]

    research_id: str
    researcher_history: Annotated[list[BaseMessage], add]


# the substate's are mainly for typing the nodes only thats why im not adding the reducers


class PreprocessorState(TypedDict):
    query: NotRequired[str]
    message_history: list[BaseMessage]


class OrchestratorState(TypedDict):
    message_history: list[BaseMessage]
    should_research: bool

    planned_subtasks: NotRequired[list[str]]

    research_id: NotRequired[str]
    research_findings: NotRequired[Sources]
    response: NotRequired[str]


class ResearcherState(TypedDict):
    planned_subtasks: NotRequired[list[str]]

    current_iteration: NotRequired[int]
    researcher_history: list[BaseMessage]
    should_continue: bool

    research_id: NotRequired[str]
    message_history: NotRequired[list[BaseMessage]]
