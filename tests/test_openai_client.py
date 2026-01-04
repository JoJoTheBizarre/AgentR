import pytest
from unittest.mock import patch, MagicMock

from client import OpenAIClient
from client.message_types import UserMessage, SystemMessage, AssistantMessage

# ---------------------------------------------------------------------------
# Helper: fake OpenAI response
# ---------------------------------------------------------------------------

def make_openai_choice(content="Hello!", tool_calls=None):
    # Mock OpenAI message object
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return MagicMock(message=msg)

def make_tool_call_response():
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call_123"
    tool_call_mock.type = "function"
    tool_call_mock.function.name = "echo"
    tool_call_mock.function.arguments = '{"text":"Hello"}'

    msg_mock = MagicMock()
    msg_mock.content = None
    msg_mock.tool_calls = [tool_call_mock]

    return MagicMock(choices=[MagicMock(message=msg_mock)])


# ---------------------------------------------------------------------------
# Tests using patch
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return OpenAIClient(api_key="fake-key", model="mock-model")


def test_patch_openai_raw_message(client):
    system_message = SystemMessage(content="System instructions")
    user_message = UserMessage(content="Say hello")

    with patch.object(client.client.chat.completions, "create") as mock_create:
        mock_choice = make_openai_choice(content="Hello from patched OpenAI!")
        mock_create.return_value = MagicMock(choices=[mock_choice])

        response: AssistantMessage = client.chat(system_message, [user_message], tools=None)

    assert isinstance(response, AssistantMessage)
    assert response.content == "Hello from patched OpenAI!"
    assert response.tool_calls is None


def test_patch_openai_tool_call(client):
    system_message = SystemMessage(content="System instructions")
    user_message = UserMessage(content="Call a tool")

    with patch.object(client.client.chat.completions, "create") as mock_create:
        tool_call_mock = MagicMock()
        tool_call_mock.id = "call_123"
        tool_call_mock.type = "function"
        tool_call_mock.function.name = "echo"
        tool_call_mock.function.arguments = '{"text":"Hello"}'

        msg_mock = MagicMock()
        msg_mock.content = None
        msg_mock.tool_calls = [tool_call_mock]

        mock_create.return_value = MagicMock(choices=[MagicMock(message=msg_mock)])

        response: AssistantMessage = client.chat(system_message, [user_message], tools=None)

    assert isinstance(response, AssistantMessage)
    assert response.content is None
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.id == "call_123"
    assert tc.function.name == "echo"
    assert tc.function.arguments == '{"text":"Hello"}'
