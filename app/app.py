import logging
import time

import typer

from src.client import OpenAIClient
from src.config import EnvConfig
from src.graph.agent import AgentR

logger = logging.getLogger(__name__)

app = typer.Typer()


#factory to create the agent
def _init_agent(tracing: bool) -> AgentR:
    env_config = EnvConfig()  # type: ignore
    logger.info(f"Initializing AgentR with {env_config.model_name}")

    if tracing:
        logger.info("Langfuse tracing enabled")
    else:
        logger.debug("Langfuse tracing disabled")

    client = OpenAIClient(client_config=env_config)
    agent = AgentR(client, env_config, tracing)
    logger.debug("Agent initialization complete")
    return agent


@app.command()
def complete(query: str, tracing: bool):
    start_time = time.time()
    logger.info(f"Processing query ({len(query)} chars)")

    try:
        agent = _init_agent(tracing)
        response = agent.invoke(request=query)
        duration = time.time() - start_time
        logger.info(f"Query completed successfully in {duration:.2f}s")
        print(response)
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Query failed after {duration:.2f}s: {type(e).__name__}")
        raise


if __name__ == "__main__":
    app()
