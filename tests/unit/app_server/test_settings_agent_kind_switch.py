"""Unit tests for ``Settings.update`` agent-kind switch behaviour.

The discriminated ``OpenHandsAgentSettings | ACPAgentSettings`` union means a
naive deep-merge of the incoming kind's fields onto the outgoing kind's dump
produces a mongrel (e.g. ``llm`` plus ``acp_command``) that fails validation
and 500s the settings endpoint. The fix is to start from a fresh base for
the new kind.

This PR ships the minimum-viable switch — the new kind comes up at defaults.
Cross-kind config preservation (snapshot/restore in ``saved_agent_configs``)
is tracked as a follow-up.
"""

from __future__ import annotations

from openhands.app_server.settings.settings_models import Settings


def _set_acp(
    command: list[str] | None = None,
    *,
    acp_env: dict | None = None,
) -> dict:
    return {
        'agent_settings_diff': {
            'agent_kind': 'acp',
            'acp_command': command
            or ['npx', '-y', '@agentclientprotocol/claude-agent-acp'],
            'acp_args': [],
            **({'acp_env': acp_env} if acp_env is not None else {}),
        }
    }


def _set_openhands(
    *,
    llm_model: str | None = None,
    mcp_config: dict | None = None,
) -> dict:
    diff: dict = {'agent_kind': 'openhands'}
    if llm_model is not None:
        diff['llm'] = {'model': llm_model}
    if mcp_config is not None:
        diff['mcp_config'] = mcp_config
    return {'agent_settings_diff': diff}


def test_kind_switch_does_not_raise():
    """OH → ACP → OH must not 500.

    Regression guard for the discriminated-union mongrel: deep-merging the
    OH dump onto an ``acp_command`` payload would produce a dict carrying
    both ``llm`` and ``acp_command``, which neither branch of
    ``AgentSettingsConfig`` accepts.
    """
    s = Settings()
    s.update(_set_openhands(llm_model='anthropic/claude-sonnet-4-5'))

    s.update(_set_acp())
    assert s.agent_settings.agent_kind == 'acp'

    s.update(_set_openhands())
    assert s.agent_settings.agent_kind == 'openhands'


def test_kind_switch_resets_new_kind_to_defaults():
    """Switching to a new kind starts from a fresh base.

    The user's outgoing-kind config is intentionally not carried into the
    new kind — preserving it across switches is the follow-up feature.
    """
    s = Settings()
    s.update(_set_openhands(llm_model='anthropic/claude-sonnet-4-5'))

    s.update(_set_acp())

    # ACP base — ``llm`` defaults to the ACP sentinel, not the OH model.
    assert s.agent_settings.agent_kind == 'acp'
    assert s.agent_settings.llm.model != 'anthropic/claude-sonnet-4-5'


def test_acp_env_replaced_wholesale():
    """``acp_env`` is replaced wholesale (not deep-merged) so removed keys
    don't leak across saves.

    This is independent of the kind switch: any ``acp_env`` in the update
    payload replaces the stored dict in full.
    """
    s = Settings()
    s.update(_set_acp(acp_env={'FOO': '1', 'BAR': '2'}))
    assert s.agent_settings.acp_env == {'FOO': '1', 'BAR': '2'}

    s.update({'agent_settings_diff': {'acp_env': {'FOO': '9'}}})
    assert s.agent_settings.acp_env == {'FOO': '9'}


def test_kind_switch_with_inline_field_override():
    """An ``agent_kind`` switch alongside other fields in the same payload
    must apply those fields on top of the fresh base.

    e.g. switching to OH and setting an LLM model in one call: the LLM
    override must land on the fresh OH base.
    """
    s = Settings()
    s.update(_set_acp())

    s.update(_set_openhands(llm_model='model-c'))
    assert s.agent_settings.agent_kind == 'openhands'
    assert s.agent_settings.llm.model == 'model-c'


def test_replace_mcp_config_in_kind_switch():
    """``mcp_config`` replace-wholesale also works alongside a kind switch."""
    s = Settings()
    s.update(_set_acp())

    s.update(_set_openhands(mcp_config={'mcpServers': {'foo': {'command': 'foo-bin'}}}))
    assert s.agent_settings.mcp_config is not None
    assert 'foo' in s.agent_settings.mcp_config.mcpServers
