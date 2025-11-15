"""Minimal tests: mcp.json overrides persisted agent MCP servers."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest
from pydantic import SecretStr

from openhands.sdk import Agent, LLM
from openhands_cli.locations import MCP_CONFIG_FILE, AGENT_SETTINGS_PATH
from openhands_cli.tui.settings.store import AgentStore


# ---------------------- tiny helpers ----------------------

def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj))


def write_agent(root: Path, agent: Agent) -> None:
    (root / AGENT_SETTINGS_PATH).write_text(
        agent.model_dump_json(context={"expose_secrets": True})
    )


# ---------------------- fixtures ----------------------

@pytest.fixture
def persistence_dir(tmp_path, monkeypatch) -> Path:
    # Create root dir and point AgentStore at it
    root = tmp_path / "openhands"
    root.mkdir()
    monkeypatch.setattr("openhands_cli.tui.settings.store.PERSISTENCE_DIR", str(root))
    return root


@pytest.fixture
def agent_store() -> AgentStore:
    return AgentStore()


# ---------------------- tests ----------------------

@patch("openhands_cli.tui.settings.store.get_default_tools", return_value=[])
@patch("openhands_cli.tui.settings.store.get_llm_extra_body", return_value={})
def test_load_overrides_persisted_mcp_with_mcp_json_file(
    mock_meta,
    mock_tools,
    persistence_dir,
    agent_store
):
    """If agent has MCP servers, mcp.json must replace them entirely."""
    # Persist an agent that already contains MCP servers
    persisted_agent = Agent(
        llm=LLM(model="gpt-4", api_key=SecretStr("k"), usage_id="svc"),
        tools=[],
        mcp_config={
            "mcpServers": {
                "persistent_server": {"command": "python", "args": ["-m", "old_server"]}
            }
        },
    )
    write_agent(persistence_dir, persisted_agent)

    # Create mcp.json with different servers (this must fully override)
    write_json(
        persistence_dir / MCP_CONFIG_FILE,
        {
            "mcpServers": {
                "file_server": {"command": "uvx", "args": ["mcp-server-fetch"]}
            }
        },
    )

    loaded = agent_store.load()
    assert loaded is not None
    # Expect ONLY the MCP json file's config
    assert loaded.mcp_config == {
        "mcpServers": {
            "file_server": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "env": {},
                "transport": "stdio",
            }
        }
    }


@patch("openhands_cli.tui.settings.store.get_default_tools", return_value=[])
@patch("openhands_cli.tui.settings.store.get_llm_extra_body", return_value={})
def test_load_when_mcp_file_missing_ignores_persisted_mcp(
    mock_meta,
    mock_tools,
    persistence_dir,
    agent_store
):
    """If mcp.json is absent, loaded agent.mcp_config should be empty (persisted MCP ignored)."""
    persisted_agent = Agent(
        llm=LLM(model="gpt-4", api_key=SecretStr("k"), usage_id="svc"),
        tools=[],
        mcp_config={
            "mcpServers": {
                "persistent_server": {"command": "python", "args": ["-m", "old_server"]}
            }
        },
    )
    write_agent(persistence_dir, persisted_agent)

    # No mcp.json created

    loaded = agent_store.load()
    assert loaded is not None
    assert loaded.mcp_config == {}  # persisted MCP is ignored if file is missin
