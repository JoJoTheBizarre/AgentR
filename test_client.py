"""Test script for AnthropicClient."""

import asyncio
import os
from dotenv import load_dotenv

from src.agentr import AnthropicClient


def test_sync_chat():
    """Test synchronous chat."""
    load_dotenv(".env.dev")

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        {"role": "user", "content": "Say hello in one sentence."}
    ]

    print("Testing synchronous chat...")
    response = client.chat(messages)
    print(f"Response: {response['content'][0]['text']}\n")


def test_sync_stream():
    """Test synchronous streaming."""
    load_dotenv(".env.dev")

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        {"role": "user", "content": "Count from 1 to 5."}
    ]

    print("Testing synchronous streaming...")
    for chunk in client.stream(messages):
        if chunk.get("type") == "content_block_delta":
            if "delta" in chunk and "text" in chunk["delta"]:
                print(chunk["delta"]["text"], end="", flush=True)
    print("\n")


async def test_async_chat():
    """Test asynchronous chat."""
    load_dotenv(".env.dev")

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        {"role": "user", "content": "Say goodbye in one sentence."}
    ]

    print("Testing asynchronous chat...")
    response = await client.achat(messages)
    print(f"Response: {response['content'][0]['text']}\n")


async def test_async_stream():
    """Test asynchronous streaming."""
    load_dotenv(".env.dev")

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        {"role": "user", "content": "List three colors."}
    ]

    print("Testing asynchronous streaming...")
    async for chunk in client.astream(messages):
        if chunk.get("type") == "content_block_delta":
            if "delta" in chunk and "text" in chunk["delta"]:
                print(chunk["delta"]["text"], end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # Test synchronous methods
    test_sync_chat()
    test_sync_stream()

    # Test asynchronous methods
    asyncio.run(test_async_chat())
    asyncio.run(test_async_stream())

    print("All tests completed!")
