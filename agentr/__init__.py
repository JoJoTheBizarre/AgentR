"""AgentR - AI Research Assistant Framework"""

__version__ = "0.1.0"

from .agent import AgentR
from .client import OpenAIClient
from .config import EnvConfig

__all__ = ["AgentR", "OpenAIClient", "EnvConfig"]
