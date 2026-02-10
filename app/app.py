import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.markdown import Markdown
from rich.table import Table

from src.client import OpenAIClient
from src.config import EnvConfig
from src.graph.agent import AgentR
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="AgentR - AI Research Assistant",
    add_completion=False,
)
console = Console()


def _init_agent(tracing: bool, env_config: EnvConfig | None = None) -> AgentR:
    """Initialize the AgentR instance."""
    if env_config is None:
        env_config = EnvConfig()  # type: ignore

    logger.info(f"Initializing AgentR with {env_config.model_name}")

    if tracing:
        logger.info("Langfuse tracing enabled")
    else:
        logger.debug("Langfuse tracing disabled")

    client = OpenAIClient(client_config=env_config)
    agent = AgentR(client, env_config)
    logger.debug("Agent initialization complete")
    return agent


def _display_response(response: str, show_markdown: bool = True):
    """Display the agent response with rich formatting."""
    if show_markdown:
        console.print(
            Panel(
                Markdown(response),
                title="[bold cyan]Response[/bold cyan]",
                border_style="cyan",
            )
        )
    else:
        console.print(
            Panel(
                response, title="[bold cyan]Response[/bold cyan]", border_style="cyan"
            )
        )


def _display_error(error: Exception, duration: float):
    """Display error message with rich formatting."""
    console.print(f"\n[bold red]Error:[/bold red] {type(error).__name__}")
    console.print(f"[red]{str(error)}[/red]")
    console.print(f"[dim]Failed after {duration:.2f}s[/dim]\n")


@app.command()
def complete(
    query: str = typer.Argument(..., help="The query to process"),
    tracing: bool = typer.Option(
        False, "--tracing", "-t", help="Enable Langfuse tracing"
    ),
    markdown: bool = typer.Option(
        True, "--markdown/--no-markdown", "-m/-M", help="Render response as markdown"
    ),
):
    """
    Process a single query and return the response.

    Example:
        uv run main.py complete "What are the latest AI trends?"
    """
    env_config = EnvConfig()  # type: ignore
    setup_logging(env_config)

    console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")

    start_time = time.time()
    logger.info(f"Processing query ({len(query)} chars)")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Processing...", total=None)
            agent = _init_agent(tracing, env_config)
            response = agent.invoke(request=query)

        duration = time.time() - start_time
        logger.info(f"Query completed successfully in {duration:.2f}s")

        _display_response(response, markdown)
        console.print(f"\n[dim]Completed in {duration:.2f}s[/dim]\n")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Query failed after {duration:.2f}s: {type(e).__name__}")
        _display_error(e, duration)
        raise typer.Exit(code=1)


@app.command()
def chat(
    tracing: bool = typer.Option(
        False, "--tracing", "-t", help="Enable Langfuse tracing"
    ),
    markdown: bool = typer.Option(
        True, "--markdown/--no-markdown", "-m/-M", help="Render responses as markdown"
    ),
):
    """
    Start an interactive chat session with the agent.

    Example:
        uv run main.py chat
        uv run main.py chat --tracing
    """
    env_config = EnvConfig()  # type: ignore
    setup_logging(env_config)

    # Display welcome message
    console.print(
        Panel.fit(
            "[bold cyan]AgentR Interactive Chat[/bold cyan]\n"
            "Type your queries and press Enter. Type 'exit' or 'quit' to end the session.\n"
            "Commands: /help, /info, /clear",
            border_style="cyan",
        )
    )

    agent = _init_agent(tracing, env_config)
    query_count = 0

    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold green]You[/bold green]")

            if not query.strip():
                continue

            # Handle commands
            if query.lower() in ["exit", "quit", "/exit", "/quit"]:
                if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                    console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                    break
                continue

            if query.lower() in ["/help", "help"]:
                _display_help()
                continue

            if query.lower() in ["/info", "info"]:
                _display_info(env_config, tracing, query_count)
                continue

            if query.lower() in ["/clear", "clear"]:
                console.clear()
                continue

            query_count += 1
            start_time = time.time()

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task(description="Thinking...", total=None)
                    response = agent.invoke(request=query)

                duration = time.time() - start_time

                console.print(f"\n[bold cyan]AgentR[/bold cyan]:")
                _display_response(response, markdown)
                console.print(f"[dim]Response time: {duration:.2f}s[/dim]")

            except Exception as e:
                duration = time.time() - start_time
                _display_error(e, duration)

        except KeyboardInterrupt:
            if Confirm.ask("\n\n[yellow]Exit chat?[/yellow]"):
                console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                break
            continue
        except EOFError:
            console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
            break


def _display_help():
    """Display help information."""
    help_table = Table(
        title="Available Commands", show_header=True, header_style="bold cyan"
    )
    help_table.add_column("Command", style="green")
    help_table.add_column("Description")

    help_table.add_row("exit, quit", "Exit the chat session")
    help_table.add_row("/help", "Show this help message")
    help_table.add_row("/info", "Show session information")
    help_table.add_row("/clear", "Clear the screen")

    console.print(help_table)


def _display_info(env_config: EnvConfig, tracing: bool, query_count: int):
    """Display session information."""
    info_table = Table(
        title="Session Information", show_header=True, header_style="bold cyan"
    )
    info_table.add_column("Setting", style="green")
    info_table.add_column("Value")

    info_table.add_row("Model", env_config.model_name)
    info_table.add_row("Tracing", "Enabled" if tracing else "Disabled")
    info_table.add_row("Queries Processed", str(query_count))

    console.print(info_table)


@app.command()
def batch(
    input_file: Path = typer.Argument(
        ..., help="Path to file with queries (one per line)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for responses"
    ),
    tracing: bool = typer.Option(
        False, "--tracing", "-t", help="Enable Langfuse tracing"
    ),
):
    """
    Process multiple queries from a file.

    Example:
        uv run main.py batch queries.txt
        uv run main.py batch queries.txt -o responses.txt
    """
    env_config = EnvConfig()  # type: ignore
    setup_logging(env_config)

    if not input_file.exists():
        console.print(f"[bold red]Error:[/bold red] File '{input_file}' not found")
        raise typer.Exit(code=1)

    # Read queries
    with open(input_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        console.print("[yellow]No queries found in file[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]Processing {len(queries)} queries...[/bold blue]\n")

    agent = _init_agent(tracing, env_config)
    responses = []

    for i, query in enumerate(queries, 1):
        console.print(f"[cyan]Query {i}/{len(queries)}:[/cyan] {query}")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description=f"Processing query {i}...", total=None)
                response = agent.invoke(request=query)

            responses.append(f"Query: {query}\n\nResponse:\n{response}\n\n{'=' * 80}\n")
            console.print("[green]✓[/green] Complete\n")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed: {type(e).__name__}\n")
            responses.append(
                f"Query: {query}\n\nError: {type(e).__name__}: {str(e)}\n\n{'=' * 80}\n"
            )

    # Save responses
    if output_file:
        with open(output_file, "w") as f:
            f.writelines(responses)
        console.print(f"\n[bold green]Responses saved to {output_file}[/bold green]\n")
    else:
        for response in responses:
            console.print(response)


@app.command()
def info():
    """Display information about the AgentR configuration."""
    env_config = EnvConfig()  # type: ignore

    info_panel = Table(
        title="AgentR Configuration", show_header=True, header_style="bold cyan"
    )
    info_panel.add_column("Setting", style="green")
    info_panel.add_column("Value")

    info_panel.add_row("Model", env_config.model_name)
    info_panel.add_row("Environment", env_config.environment)

    console.print(info_panel)


if __name__ == "__main__":
    app()
