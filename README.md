# AgentR 🤖

**Research Agent Using a Plan-and-Execute Architecture**

AgentR is a stateful research agent built with **LangGraph**.
Inspired by the **[Plan-and-Solve](https://arxiv.org/abs/2305.04091)** paper and the
**[BabyAGI](https://github.com/yoheinakajima/babyagi)** project, it dynamically plans,
executes, and refines research tasks through a flexible tool system.

## 🚀 Features

- **Intelligent Research Decision**: An orchestrator determines whether a query requires research
- **Multi-Step Planning**: Generates and refines execution plans iteratively
- **Dynamic Tool Selection**: Type-safe tools with runtime configuration
- **Iterative Research**: Multiple search cycles with quality control and source tracking
- **Observability**: Integrated Langfuse tracing and monitoring
- **Modular Design**: Extensible node- and tool-based architecture

## 🏗️ Architecture

AgentR follows a **Plan-and-Execute** workflow:
1. Generate a multi-step research plan
2. Execute each step using appropriate tools
3. Re-evaluate and adapt the plan as new information is gathered

This workflow is implemented as a stateful, graph-based system using LangGraph.


### Computational Graph
![graph_diagram](graph_diagram.png)

### Core Components

1. **AgentR Graph (`src/graph/agent.py`)**: Main stateful graph built with LangGraph
2. **Nodes**:
   - **Preprocessor**: Processes user queries into message history
   - **Orchestrator**: Decides if research is needed and synthesizes results
   - **Researcher**: Conducts iterative web research with dynamic tool access
   - **Tool Node**: Executes tools requested by the researcher
4. **Tool System (`src/tools/`)**: Registry pattern with extensible base tools
5. **LLM Client (`src/client/`)**: OpenAI-compatible client wrapper

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- API keys for:
  - OpenAI/DeepSeek (required)
  - Tavily (recommended for web search)
  - Langfuse (optional for observability)