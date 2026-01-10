from enum import StrEnum
from typing import NotRequired

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

    message_history: list[BaseMessage]

    should_research: bool
    current_iteration: int
    max_iteration: int

    planned_subtasks: list[str]
    completed_subtasks: list[str]

    research_id: str
    research_findings: Sources


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
    active_subtasks: list[str]

    current_iteration: int

    research_id: str
    research_findings: NotRequired[Sources]
