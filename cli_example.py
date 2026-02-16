import time
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from agentr import AgentR, EnvConfig, OpenAIClient

app = typer.Typer(help="AgentR - AI Research Assistant", add_completion=False)
console = Console()


def init_agent(tracing: bool = False, thread_id: str = "default") -> AgentR:
    """Initialize AgentR instance."""
    config = EnvConfig()  # type: ignore
    client = OpenAIClient(client_config=config)
    agent = AgentR(client, config, tracing=tracing, thread_id=thread_id, enable_memory=True)
    return agent


def display_response(response: str, markdown: bool = True):
    """Display agent response."""
    if markdown:
        console.print(
            Panel(Markdown(response), title="[bold cyan]AgentR[/bold cyan]", border_style="cyan")
        )
    else:
        console.print(Panel(response, title="[bold cyan]AgentR[/bold cyan]", border_style="cyan"))


def show_commands():
    """Display available commands."""
    console.print(
        Panel(
            """[bold cyan]Commands:[/bold cyan]

• [green]/help[/green]  - Show this message
• [green]/info[/green]  - Session info
• [green]/clear[/green] - Clear screen
• [green]exit[/green]   - Exit chat""",
            title="[bold yellow]💡 Help[/bold yellow]",
            border_style="yellow",
        )
    )


def show_welcome():
    """Display welcome banner."""
    console.print(
        Panel(
            """[bold cyan]Welcome to AgentR![/bold cyan]

Your AI Research Assistant with conversation memory.

Ask me anything!""",
            title="[bold green]🤖 AgentR Chat[/bold green]",
            border_style="green",
        )
    )


def show_info(tracing: bool, query_count: int, thread_id: str):
    """Display session info."""
    config = EnvConfig()  # type: ignore
    table = Table(title="Session Info", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value")
    table.add_row("Model", config.model_name)
    table.add_row("Thread ID", thread_id)
    table.add_row("Tracing", "✓ Enabled" if tracing else "✗ Disabled")
    table.add_row("Queries", str(query_count))
    console.print(table)


@app.command()
def chat(
    tracing: bool = typer.Option(False, "--trace", "-t", help="Enable tracing"),
    thread_id: str = typer.Option("default", "--thread", help="Thread ID"),
):
    """Start interactive chat."""
    show_welcome()
    console.print()
    show_commands()

    agent = init_agent(tracing, thread_id)
    query_count = 0

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            if user_input.lower() in ["exit", "quit"]:
                if Confirm.ask("\n[yellow]Exit?[/yellow]"):
                    console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                    break
                continue

            if user_input.lower() in ["/help", "help"]:
                show_commands()
                continue

            if user_input.lower() in ["/info", "info"]:
                show_info(tracing, query_count, thread_id)
                continue

            if user_input.lower() in ["/clear", "clear"]:
                console.clear()
                show_welcome()
                console.print()
                show_commands()
                continue

            query_count += 1
            start = time.time()

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task(description="Thinking...", total=None)
                    response = agent.invoke(request=user_input)

                duration = time.time() - start
                console.print()
                display_response(response)
                console.print(f"[dim]Time: {duration:.2f}s[/dim]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

        except KeyboardInterrupt:
            if Confirm.ask("\n\n[yellow]Exit?[/yellow]"):
                console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                break
            continue
        except EOFError:
            console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
            break


@app.command()
def info():
    """Display config."""
    config = EnvConfig()  # type: ignore
    table = Table(title="AgentR Config", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value")
    table.add_row("Model", config.model_name)
    table.add_row("API URL", config.api_url)
    table.add_row("Environment", config.environment)
    console.print(table)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    tracing: bool = typer.Option(False, "--trace", "-t"),
    thread_id: str = typer.Option("default", "--thread"),
):
    """AgentR - AI Research Assistant"""
    if ctx.invoked_subcommand is None:
        chat(tracing=tracing, thread_id=thread_id)


if __name__ == "__main__":
    app()
