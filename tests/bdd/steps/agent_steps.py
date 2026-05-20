"""BDD step implementations for agent behavior.

Implements Given/When/Then steps for:
- Agent lifecycle (session creation, state transitions)
- LLM interactions (prompting, response handling)
- Tool execution (registry lookup, invocation)
- Sandbox management (command execution, filesystem operations)
"""

from __future__ import annotations

from typing import Any

from pytest_bdd import given, then, when

from tests.bdd.mocks.llm_mock import LLMMock


# ============================================================================
# Given Steps
# ============================================================================


@given("the LLM is configured with default responses")
def llm_configured_defaults(mock_llm: LLMMock, agent_context: Any) -> None:
    """Configure LLM with default response patterns.

    Args:
        mock_llm: Mock LLM instance
        agent_context: Agent context for steps
    """
    agent_context.llm = mock_llm
    # Default patterns already set up in LLMMock initialization
    # Just reset to ensure clean state
    mock_llm.reset()


@given("the LLM is configured to return a <response_type> action")
def llm_configured_action(
    mock_llm: LLMMock, response_type: str, agent_context: Any
) -> None:
    """Configure LLM to return a specific action type.

    Args:
        mock_llm: Mock LLM instance
        response_type: Action type (run, edit, think, ask_followup)
        agent_context: Agent context
    """
    agent_context.llm = mock_llm
    mock_llm.configure_response(
        trigger="test",
        action=response_type,
    )


# ============================================================================
# When Steps
# ============================================================================


@when("the user sends a message")
def user_sends_message(agent_context: Any) -> None:
    """Simulate user sending a message to agent.

    Args:
        agent_context: Agent context
    """
    if not agent_context.llm:
        raise RuntimeError("LLM not configured. Use 'Given' step first.")

    message = "test message"  # Simple message for now

    # Add user message to conversation
    user_msg = {"role": "user", "content": message}
    agent_context.messages.append(user_msg)

    # Call LLM (synchronous)
    response = agent_context.llm.call_sync(message)
    agent_context.last_response = response
    agent_context.llm_call_count += 1

    # Add assistant response
    assistant_msg = {
        "role": "assistant",
        "content": str(response),
        "action": response.get("action"),
    }
    agent_context.messages.append(assistant_msg)



# ============================================================================
# Then Steps
# ============================================================================


@then("the agent receives the message")
def agent_receives_message(agent_context: Any) -> None:
    """Verify agent received the message.

    Args:
        agent_context: Agent context
    """
    assert len(agent_context.messages) > 0, "Agent should have received message"
    assert (
        agent_context.messages[0]["role"] == "user"
    ), "First message should be from user"


@then("the LLM is called with the user message")
def llm_called_with_message(agent_context: Any) -> None:
    """Verify LLM was called.

    Args:
        agent_context: Agent context
    """
    assert agent_context.llm is not None, "LLM should be configured"
    assert agent_context.llm_call_count > 0, "LLM should have been called"


@then("the agent returns a response with an action")
def agent_returns_response_with_action(agent_context: Any) -> None:
    """Verify agent returned response with action.

    Args:
        agent_context: Agent context
    """
    assert agent_context.last_response is not None, "Agent should have returned response"
    assert (
        "action" in agent_context.last_response
    ), "Response should contain action field"
    assert agent_context.last_response["action"] in [
        "run",
        "edit",
        "think",
        "ask_followup",
    ], "Action should be valid type"


@then("the agent returns a response")
def agent_returns_response(agent_context: Any) -> None:
    """Verify agent returned any response.

    Args:
        agent_context: Agent context
    """
    assert agent_context.last_response is not None, "Agent should have returned response"




@then("the LLM has processed (\\d+) calls")
def llm_processed_n_calls(count: str, agent_context: Any) -> None:
    """Verify LLM was called expected number of times.

    Args:
        count: Expected call count (as string from regex)
        agent_context: Agent context
    """
    count_int = int(count)
    assert agent_context.llm_call_count == count_int, (
        f"Expected {count_int} LLM calls, "
        f"but got {agent_context.llm_call_count}"
    )


@then("the conversation history is preserved in memory")
def conversation_history_preserved(agent_context: Any) -> None:
    """Verify conversation history is preserved.

    Args:
        agent_context: Agent context
    """
    assert agent_context.llm is not None, "LLM should be configured"
    memory = agent_context.llm.get_memory()
    assert len(memory.get_history()) > 0, "Memory should contain conversation history"


@then("the LLM has been called twice")
def llm_called_twice(agent_context: Any) -> None:
    """Verify LLM was called twice.

    Args:
        agent_context: Agent context
    """
    assert agent_context.llm_call_count == 2, (
        f"Expected 2 calls, got {agent_context.llm_call_count}"
    )
