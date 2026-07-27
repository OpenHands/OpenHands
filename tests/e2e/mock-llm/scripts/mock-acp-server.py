"""Mock ACP (Agent Client Protocol) server for E2E tests.

A minimal stdio-based ACP agent that speaks JSON-RPC over stdin/stdout.
The agent-server spawns this as a subprocess via ``acp_command`` and
communicates with it using the ACP protocol.

The agent responds to prompts with a scripted text reply containing
``REPLY_TOKEN``, which the E2E test verifies appeared in the UI.

Usage:
    python mock-acp-server.py [--reply-token TOKEN]

Requires:
    pip install agent-client-protocol  (installed as dep of openhands-sdk)
"""

import argparse
import asyncio
import sys

import acp
from acp.schema import (
    AgentCapabilities,
    Implementation,
    PromptCapabilities,
)

REPLY_TOKEN = "MOCK_ACP_E2E_REPLY_OK"


class MockACPAgent(acp.Agent):
    """Minimal ACP agent that returns a scripted reply to every prompt."""

    def __init__(self, reply_token: str = REPLY_TOKEN) -> None:
        self.reply_token = reply_token
        self._conn: acp.Client | None = None

    def on_connect(self, conn: acp.Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities=None,
        client_info=None,
        **kwargs,
    ) -> acp.InitializeResponse:
        print("[mock-acp] initialize", file=sys.stderr, flush=True)
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(
                name="mock-acp-e2e",
                title="Mock ACP E2E Agent",
                version="1.0.0",
            ),
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories=None,
        **kwargs,
    ) -> acp.NewSessionResponse:
        print(f"[mock-acp] new_session cwd={cwd}", file=sys.stderr, flush=True)
        return acp.NewSessionResponse(session_id="mock-acp-session-001")

    async def prompt(
        self,
        prompt,
        session_id: str,
        message_id: str | None = None,
        **kwargs,
    ) -> acp.PromptResponse:
        # Extract user text for logging
        user_text = ""
        if prompt:
            for block in prompt:
                if hasattr(block, "text"):
                    user_text += block.text
        print(
            f"[mock-acp] prompt session={session_id} text={user_text!r}",
            file=sys.stderr,
            flush=True,
        )

        # Send the agent's text reply as a session/update notification
        if self._conn:
            await self._conn.session_update(
                session_id=session_id,
                update=acp.update_agent_message_text(self.reply_token),
            )

        return acp.PromptResponse(stop_reason="end_turn")


async def main(reply_token: str) -> None:
    agent = MockACPAgent(reply_token=reply_token)
    print(f"[mock-acp] starting (token={reply_token})", file=sys.stderr, flush=True)
    await acp.run_agent(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock ACP agent for E2E tests")
    parser.add_argument(
        "--reply-token",
        default=REPLY_TOKEN,
        help="Token to include in agent replies (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.reply_token))
