"""
Tests for OpenAIClient.
These tests require OPENAI_API_KEY to be set.
"""

import os
import pytest
from dotenv import load_dotenv

from src.agentr import OpenAIClient
from src.agentr.message_types import UserMessage, SystemMessage


# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

load_dotenv(".env.dev")


@pytest.fixture(scope="session")
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAIClient(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Synchronous tests
# ---------------------------------------------------------------------------

def test_sync_chat(openai_client):
    messages = [
        UserMessage(content="Say hello in one sentence.")
    ]

    response = openai_client.chat(messages)

    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["message"]["content"].strip()


def test_sync_stream(openai_client):
    messages = [
        UserMessage(content="Count from 1 to 5.")
    ]

    collected_text = ""

    for chunk in openai_client.stream(messages):
        assert "choices" in chunk
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            collected_text += delta["content"]

    assert collected_text.strip()
    assert any(char.isdigit() for char in collected_text)


# ---------------------------------------------------------------------------
# Asynchronous tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_chat(openai_client):
    messages = [
        UserMessage(content="Say goodbye in one sentence.")
    ]

    response = await openai_client.achat(messages)

    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["message"]["content"].strip()


@pytest.mark.asyncio
async def test_async_stream(openai_client):
    messages = [
        UserMessage(content="List three colors.")
    ]

    collected_text = ""

    async for chunk in openai_client.astream(messages):
        assert "choices" in chunk
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            collected_text += delta["content"]

    assert collected_text.strip()


# ---------------------------------------------------------------------------
# Compatibility tests
# ---------------------------------------------------------------------------

def test_backward_compatibility_dict_messages(openai_client):
    """Test that OpenAIClient still accepts raw dict messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."}
    ]

    response = openai_client.chat(messages)

    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["message"]["content"].strip()


def test_mixed_message_types(openai_client):
    """Test that OpenAIClient accepts mixed Message and dict types."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        UserMessage(content="Say hello.")
    ]

    response = openai_client.chat(messages)

    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["message"]["content"].strip()
