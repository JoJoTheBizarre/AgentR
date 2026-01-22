"""
State fixture generators for testing.
"""

from src.models.states import AgentState, ResearcherState


def create_agent_state(
    query: str = "Test research query",
    response: str = "",
    message_history: list | None = None,
    should_delegate: bool = False,
    should_continue: bool = False,
    current_iteration: int = 0,
    planned_subtasks: list[str] | None = None,
    sub_agent_call_id: str = "",
    researcher_history: list | None = None,
) -> AgentState:
    """Create a standardized AgentState for testing."""
    return {
        "query": query,
        "response": response,
        "message_history": message_history or [],
        "should_delegate": should_delegate,
        "should_continue": should_continue,
        "current_iteration": current_iteration,
        "planned_subtasks": planned_subtasks or [],
        "sub_agent_call_id": sub_agent_call_id,
        "researcher_history": researcher_history or [],
    }


def create_researcher_state(
    should_continue: bool = False,
    current_iteration: int = 0,
    planned_subtasks: list[str] | None = None,
    researcher_history: list | None = None,
    sub_agent_call_id: str = "test-call-id",
    message_history: list | None = None,
) -> ResearcherState:
    """Create ResearcherState for testing."""
    return {
        "should_continue": should_continue,
        "current_iteration": current_iteration,
        "planned_subtasks": planned_subtasks or ["Subtask 1", "Subtask 2"],
        "researcher_history": researcher_history or [],
        "sub_agent_call_id": sub_agent_call_id,
        "message_history": message_history or [],
    }


def create_agent_state_with_delegation(
    query: str = "Complex research query",
    planned_subtasks: list[str] | None = None,
) -> AgentState:
    """Create an AgentState where research should be delegated."""
    return create_agent_state(
        query=query,
        should_delegate=True,
        planned_subtasks=planned_subtasks
        or [
            "Research subtask 1",
            "Research subtask 2",
            "Research subtask 3",
        ],
    )


def create_agent_state_with_research_in_progress(
    current_iteration: int = 1,
    planned_subtasks: list[str] | None = None,
) -> AgentState:
    """Create an AgentState where research is in progress."""
    return create_agent_state(
        should_delegate=True,
        should_continue=True,
        current_iteration=current_iteration,
        planned_subtasks=planned_subtasks or ["Remaining subtask"],
    )
