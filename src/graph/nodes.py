from enum import StrEnum


class NodeName(StrEnum):
    """Enumeration of all node names in the AgentR graph."""

    PREPROCESSOR = "preprocessor"
    RESEARCHER = "researcher"
    ORCHESTRATOR = "orchestrator"
    TOOL_NODE = "tool_node"
