"""Exercise the mock ACP agent over its real stdio protocol boundary."""

import asyncio
from pathlib import Path
import sys

import acp
from acp.schema import AgentMessageChunk, TextContentBlock
import pytest


@pytest.mark.parametrize("reply_token", [None, "CUSTOM_ACP_REPLY_OK"])
def test_prompt_streams_reply_and_ends_turn(tmp_path, reply_token):
    class Client(acp.Client):
        def __init__(self):
            self.updates = []

        async def session_update(self, session_id, update, **kwargs):
            self.updates.append((session_id, update))

    async def run():
        client = Client()
        script = Path(__file__).with_name("mock-acp-server.py")
        args = [] if reply_token is None else ["--reply-token", reply_token]
        async with acp.spawn_agent_process(client, sys.executable, str(script), *args) as (
            conn,
            _process,
        ):
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await conn.new_session(cwd=str(tmp_path), mcp_servers=[])

            response = await conn.prompt(
                session_id=session.session_id,
                prompt=[TextContentBlock(text="Hello ACP agent, please reply.")],
            )

            assert response.stop_reason == "end_turn"
            assert len(client.updates) == 1
            session_id, update = client.updates[0]
            assert session_id == session.session_id
            assert isinstance(update, AgentMessageChunk)
            assert isinstance(update.content, TextContentBlock)
            assert update.content.text == (reply_token or "MOCK_ACP_E2E_REPLY_OK")

    asyncio.run(asyncio.wait_for(run(), timeout=10))
