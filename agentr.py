#!/usr/bin/env python3
"""
AgentR Command Line Interface

A simple entry point for interacting with the AgentR research agent system.
Supports both command-line arguments and interactive mode.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from client import OpenAIClient
from config import ClientSettings
from graph.agent import AgentR


def initialize_agent(
    api_key: str | None = None,
    model: str | None = None,
    api_url: str | None = None,
) -> AgentR:
    """Initialize the AgentR agent with configuration.

    Args:
        api_key: OpenAI API key (overrides environment variable)
        model: Model name (overrides environment variable)
        api_url: API URL (overrides environment variable)

    Returns:
        Initialized AgentR instance
    """
    try:
        client_config = ClientSettings()
        print(f"✓ Using environment configuration: model={client_config.model_name}")

        # Set Tavily API key in environment if loaded from config
        if client_config.tavily_api_key and not os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = client_config.tavily_api_key
            print("✓ Tavily API key loaded from configuration")

    except Exception as e:
        print(f"⚠  Configuration error: {e}")
        print("\nConfiguration options:")
        print("1. Environment variables:")
        print("   - OPENAI_API_KEY (required)")
        print("   - OPENAI_API_URL (optional)")
        print("   - OPENAI_MODEL_NAME (optional, defaults to 'gpt-4o-mini')")
        print("   - TAVILY_API_KEY (optional but recommended)")
        print("\n2. Command-line arguments:")
        print("   --api-key KEY    Override API key")
        print("   --model MODEL    Override model name")
        print("   --api-url URL    Override API URL")
        print("\n3. .env file:")
        print("   Create a .env.dev file (or .env) in the project root")
        print("   with the above variables")
        sys.exit(1)

    try:
        # Initialize OpenAI client
        client = OpenAIClient(client_config=client_config)

        # Initialize AgentR
        agent = AgentR(llm_client=client)
        print("✓ AgentR initialized successfully")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize AgentR: {e}")
        if "TAVILY_API_KEY" in str(e):
            print("\nℹ️  Note: Tavily API key is not set.")
            print("Web search functionality will be limited.")
            print(
                "Set TAVILY_API_KEY environment variable for full research capabilities."
            )
        sys.exit(1)


def run_single_query(agent: AgentR, query: str) -> None:
    """Run a single query through the agent and print the response."""
    print(f"\n🔍 Query: {query}")
    print("-" * 80)

    try:
        response = agent.invoke(query)
        print(f"\n📝 Response:\n{response}")
    except Exception as e:
        print(f"\n✗ Error processing query: {e}")
        if "TAVILY_API_KEY" in str(e):
            print("\nℹ️  Note: Tavily API key required for web search.")
            print("Set TAVILY_API_KEY environment variable for research capabilities.")


def interactive_mode(agent: AgentR) -> None:
    """Run in interactive mode, accepting multiple queries."""
    print("\n" + "=" * 80)
    print("🤖 AgentR Interactive Mode")
    print("Enter queries (type 'quit', 'exit', or 'q' to exit)")
    print("=" * 80)

    while True:
        try:
            query = input("\n💬 Query: ").strip()

            if query.lower() in ("quit", "exit", "q", ""):
                print("👋 Goodbye!")
                break

            run_single_query(agent, query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AgentR - AI research agent for answering queries with web research capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is the capital of France?"
  %(prog)s -i
  %(prog)s --query "Latest developments in quantum computing"

Environment Variables:
  OPENAI_API_KEY      OpenAI API key (required)
  OPENAI_API_URL      OpenAI API URL (optional)
  OPENAI_MODEL_NAME   Model name (default: gpt-4o-mini)
  TAVILY_API_KEY      Tavily API key for web search (optional but recommended)
        """,
    )

    parser.add_argument(
        "query", nargs="?", help="Query to process (omit for interactive mode)"
    )

    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument(
        "--query",
        dest="explicit_query",
        help="Explicit query (alternative to positional argument)",
    )

    parser.add_argument("--model", help="Override model name")

    parser.add_argument("--api-key", help="Override OpenAI API key")

    parser.add_argument("--api-url", help="Override OpenAI API URL")

    args = parser.parse_args()

    # Determine the query
    query = args.explicit_query or args.query

    # Check if we should run in interactive mode
    if args.interactive or (not query and not args.explicit_query):
        agent = initialize_agent(
            api_key=args.api_key,
            model=args.model,
            api_url=args.api_url,
        )
        interactive_mode(agent)
    else:
        if not query:
            print("✗ Error: No query provided")
            parser.print_help()
            sys.exit(1)

        agent = initialize_agent(
            api_key=args.api_key,
            model=args.model,
            api_url=args.api_url,
        )
        run_single_query(agent, query)


if __name__ == "__main__":
    main()
