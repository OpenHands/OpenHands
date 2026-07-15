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

from openhands.app_server.settings.settings_models import (
    Settings,
    _load_persisted_agent_settings,
)
from openhands.sdk.settings.model import AGENT_SETTINGS_SCHEMA_VERSION


def _set_acp(
    command: list[str] | None = None,
) -> dict:
    return {
        'agent_settings_diff': {
            'agent_kind': 'acp',
            'acp_command': command
            or ['npx', '-y', '@agentclientprotocol/claude-agent-acp'],
            'acp_args': [],
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
    assert 'foo' in s.agent_settings.mcp_config


def test_loader_normalizes_legacy_llm_tag_at_current_schema_version():
    """A persisted ``agent_kind: 'llm'`` row already at the current
    ``schema_version`` must read back as ``openhands``.

    The SDK's ``llm -> openhands`` rename only fires while advancing the
    schema version, so an ``'llm'`` payload already at the current version is
    not migrated and would otherwise validate as the deprecated
    ``LLMAgentSettings`` (``agent_kind == 'llm'``). The loader normalizes it so
    every read stays on the canonical ``{openhands, acp}`` variants — this is
    the one legitimate job the deleted force-cast used to do.
    """
    loaded = _load_persisted_agent_settings(
        {
            'agent_kind': 'llm',
            'schema_version': AGENT_SETTINGS_SCHEMA_VERSION,
            'llm': {'model': 'anthropic/claude-sonnet-4-5'},
        }
    )

    assert loaded.agent_kind == 'openhands'
    assert loaded.llm.model == 'anthropic/claude-sonnet-4-5'


def test_loader_preserves_acp_variant_without_coercion():
    """The loader must leave ``agent_kind: 'acp'`` alone — the ``llm``
    normalization must not regress into the cross-variant coercion that 500'd
    ACP settings (``ACPAgentSettings.agent_context`` is nullable; the OpenHands
    shape rejects ``None``).
    """
    loaded = _load_persisted_agent_settings(
        {
            'agent_kind': 'acp',
            'acp_server': 'claude-code',
            'llm': {'model': 'litellm_proxy/anthropic/claude-sonnet-4'},
        }
    )

    assert loaded.agent_kind == 'acp'
    assert loaded.agent_context is None
