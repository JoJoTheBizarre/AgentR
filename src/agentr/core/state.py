from enum import StrEnum
from typing import Any, NotRequired, TypedDict

from .messages import Message


class NodeType(StrEnum):
    RESEARCH = "research"
    TOOL = "tool"
    SYNTHESIZE = "synthesize"
    ORCHESTRATOR = "orchestrator"


class AgentState(TypedDict, total=False):
    """State container for the research agent.

    Attributes:
        query:
            The original research query that the agent is expected to answer.

        message_history:
            Conversation history between the user and the orchestrator,
            including system, user, assistant, and tool messages.
            This history is not shared with the researcher.

        research_history:
            Accumulated research findings produced by the researcher,
            including extracted information and source metadata.
            This history is shared only between the orchestrator and researcher.

        response:
            The agent's final or intermediate response to the user's query.

        current_iteration:
            The current iteration index within the research loop.

        max_iterations:
            The maximum number of research iterations allowed before stopping.

        should_synthesize:
            Flag indicating whether research is complete and control
            should be routed to the synthesizer.

        should_research:
            Flag indicating whether control should be routed to the researcher
            for additional research steps.

        subtasks:
            A list of structured research subtasks assigned to the researcher.
    """

    query: str
    message_history: list[Message]
    research_history: list[dict[str, Any]]
    response: NotRequired[str]
    current_iteration: int
    max_iterations: int
    should_synthesize: bool
    should_research: bool
    subtasks: list[dict[str, Any]]
