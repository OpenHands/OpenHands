"""Mock ACP (Agent Client Protocol) server for E2E tests.

A minimal stdio-based ACP agent that speaks JSON-RPC over stdin/stdout.
The agent-server spawns this as a subprocess via ``acp_command`` and
communicates with it using the ACP protocol.

The agent responds to prompts with a scripted text reply containing
``REPLY_TOKEN``, which the E2E test verifies appeared in the UI.

M6a: the agent also advertises two ``session/new`` ``configOptions``
selects — "model" (category "model") and "effort" (category
"thought_level") — mirroring claude-agent-acp 0.44+'s config-option model
selection mechanism (see openhands-sdk's
``openhands/sdk/agent/acp_agent.py``). The stock agent-server extracts
these into ``ConversationInfo.available_models``/``current_model_id`` and
``current_effort``/``available_efforts``, which is what the Canvas chat
model pill (dynamic model + effort switching) renders and drives through
``session/set_config_option``. This is purely additive: every field the
pre-M6a spec (``mock-llm-acp-agent.spec.ts``) relies on is unchanged.

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
    ConfigOptionUpdate,
    Implementation,
    PromptCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

REPLY_TOKEN = "MOCK_ACP_E2E_REPLY_OK"

# ── M6a: model + effort configOptions ───────────────────────────────────
#
# Config-option ids mirror openhands-sdk's own constants
# (``_MODEL_CONFIG_OPTION_ID`` = "model", ``_EFFORT_CONFIG_OPTION_IDS``
# includes "effort") so the agent-server's extraction helpers
# (``_extract_session_models`` / ``_extract_session_efforts``) and its
# runtime model-switch splitter (``_apply_acp_model`` /
# ``_claude_model_config_options``) exercise the exact code paths a real
# claude-agent-acp session would.
MODEL_CONFIG_ID = "model"
EFFORT_CONFIG_ID = "effort"

MOCK_MODELS = (
    {
        "value": "mock-fast",
        "name": "Mock Fast",
        "description": "Fast, low-cost mock model",
    },
    {
        "value": "mock-smart",
        "name": "Mock Smart",
        "description": "Balanced mock model",
    },
    {
        "value": "mock-deep",
        "name": "Mock Deep",
        "description": "Deep-reasoning mock model",
    },
)
MOCK_MODEL_IDS = tuple(model["value"] for model in MOCK_MODELS)
DEFAULT_MODEL = "mock-smart"

# Mirrors claude-agent-acp's "effort" configOptions select (thought_level
# category — see openhands-sdk's ``_CLAUDE_EFFORT_LEVELS``). "default" is a
# real selectable value (Canvas's own UI-only "no suffix" sentinel), kept
# here so the option list matches what a live claude-agent-acp session
# advertises.
EFFORT_LEVELS = ("default", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "high"


def _model_option(current_value: str) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=MODEL_CONFIG_ID,
        name="Model",
        type="select",
        category="model",
        current_value=current_value,
        options=[
            SessionConfigSelectOption(
                value=model["value"],
                name=model["name"],
                description=model["description"],
            )
            for model in MOCK_MODELS
        ],
    )


def _effort_option(current_value: str) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=EFFORT_CONFIG_ID,
        name="Thinking Effort",
        type="select",
        category="thought_level",
        current_value=current_value,
        options=[
            SessionConfigSelectOption(value=level, name=level.capitalize())
            for level in EFFORT_LEVELS
        ],
    )


class MockACPAgent(acp.Agent):
    """Minimal ACP agent that returns a scripted reply to every prompt."""

    def __init__(self, reply_token: str = REPLY_TOKEN) -> None:
        self.reply_token = reply_token
        self._conn: acp.Client | None = None
        # session_id -> {"model": <current model id>, "effort": <current
        # effort level>}. The mock only ever serves one live session per
        # subprocess (the agent-server spawns a fresh subprocess per ACP
        # conversation), but keying by session id keeps this correct if
        # that ever changes (e.g. a future ``load_session``/resume test).
        self._sessions: dict[str, dict[str, str]] = {}

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
                # Contains "claude-agent" so the agent-server's
                # ``detect_acp_provider_by_agent_name`` (matched against
                # ``ACPProviderInfo.agent_name_patterns``) identifies this
                # session as the "claude-code" provider — the same
                # detection a real claude-agent-acp subprocess triggers.
                # That in turn makes ``_apply_acp_model`` actually split a
                # composite Canvas model id like "mock-smart/max" into two
                # ``session/set_config_option`` calls (config "model" then
                # config "effort") instead of sending it whole as a single
                # unrecognized "model" value.
                name="claude-agent-acp (mock)",
                title="Mock ACP E2E Agent",
                version="1.0.0",
            ),
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(),
            ),
        )

    def _session_state(self, session_id: str) -> dict[str, str]:
        return self._sessions.setdefault(
            session_id, {"model": DEFAULT_MODEL, "effort": DEFAULT_EFFORT}
        )

    def _config_options(self, session_id: str) -> list[SessionConfigOptionSelect]:
        state = self._session_state(session_id)
        return [_model_option(state["model"]), _effort_option(state["effort"])]

    async def new_session(
        self,
        cwd: str,
        additional_directories=None,
        **kwargs,
    ) -> acp.NewSessionResponse:
        session_id = "mock-acp-session-001"
        print(f"[mock-acp] new_session cwd={cwd}", file=sys.stderr, flush=True)
        # Seed default state before advertising config_options so the
        # response's "currentValue"s are well-defined even if a client asks
        # for them before any set_config_option call.
        self._session_state(session_id)
        return acp.NewSessionResponse(
            session_id=session_id,
            config_options=self._config_options(session_id),
        )

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value,
        **kwargs,
    ) -> SetSessionConfigOptionResponse:
        print(
            f"[mock-acp] set_config_option session={session_id} "
            f"config_id={config_id!r} value={value!r}",
            file=sys.stderr,
            flush=True,
        )
        state = self._session_state(session_id)
        if config_id == MODEL_CONFIG_ID:
            if not isinstance(value, str) or value not in MOCK_MODEL_IDS:
                raise acp.RequestError.invalid_params(
                    {"configId": config_id, "value": value}
                )
            state["model"] = value
        elif config_id in (EFFORT_CONFIG_ID, "reasoning_effort"):
            if not isinstance(value, str) or value not in EFFORT_LEVELS:
                raise acp.RequestError.invalid_params(
                    {"configId": config_id, "value": value}
                )
            state["effort"] = value
        else:
            # Unknown configId: tolerate rather than fail the switch. The
            # agent-server only ever sends ids it discovered on this same
            # session's own configOptions (see the SDK's
            # ``_model_config_option``/``_effort_config_option``), so this
            # branch is defensive only — it responds with the unchanged
            # full state instead of erroring out the whole switch.
            print(
                f"[mock-acp] set_config_option: unrecognized configId "
                f"{config_id!r}, leaving state unchanged",
                file=sys.stderr,
                flush=True,
            )

        options = self._config_options(session_id)
        if self._conn:
            # Optional per the ACP spec: an agent MAY push a
            # "config_option_update" session/update notification carrying
            # the full configOptions state after a set_config_option call.
            # The stock agent-server does not consume this notification
            # today (state is refreshed from the request/response round
            # trip and the next ConversationInfo poll) — sending it anyway
            # exercises that tolerance and costs nothing.
            await self._conn.session_update(
                session_id=session_id,
                update=ConfigOptionUpdate(
                    session_update="config_option_update", config_options=options
                ),
            )
        return SetSessionConfigOptionResponse(config_options=options)

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
