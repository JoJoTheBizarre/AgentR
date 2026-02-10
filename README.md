# AgentR 🤖

**Research Agent Using a Plan-and-Execute Architecture**

AgentR is a stateful research agent built with **LangGraph**.
Inspired by the **[Plan-and-Solve](https://arxiv.org/abs/2305.04091)** paper and the
**[BabyAGI](https://github.com/yoheinakajima/babyagi)** project, it dynamically plans,
executes, and refines research tasks through a flexible tool system.

## Quick Start

1. **Install uv** (if not already installed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Copy environment file**: `cp .env.dev.example .env.dev`
3. **Fill in your API keys** in `.env.dev`
4. **Install dependencies**: `uv sync`
5. **Run the agent**: `uv run main.py chat` or `uv run main.py complete "your query here"`

## Usage Examples
- Interactive chat: `uv run main.py chat`
- Single query: `uv run main.py complete "What are AI trends?"`
- See all commands: `uv run main.py --help`