# F04 — LLM, provider connection and agent profiles

Configure which agent/model a new or existing conversation uses. Source:
[LLM profiles](../../../../src/components/features/settings/llm-profiles),
[LLM route](../../../../src/routes/llm-settings.tsx),
[agent profiles](../../../../src/routes/agent-profiles-settings.tsx),
[ACP guide](../../../../docs/ACP_AGENTS.md).

## Sub-features

- `F04.llm`: list/add/edit/rename/delete/default LLM profiles; Basic/All field modes.
- `F04.connection`: shared provider connection selection, editing and key rotation.
- `F04.agent`: agent profile create/edit/default and settings route aliases.
- `F04.acp`: installed external agent, auth-required state and real conversation.
- `F04.switch`: composer/default selection and existing-conversation model switch.

## How to get to it (user POV)

Settings → LLM or Agent; direct links `/settings/llm`, `/settings/agent` and
`/settings/agents`. Also use the conversation composer profile selector, not only
the settings list. Verify aliases resolve rather than assuming they are identical.

## Driving it with Playwright

Preconditions: dummy data for form appearance only; an authorized test provider and
budget for save/validation and real conversations. ACP additionally needs the real
installed agent and its approved credentials; a mock ACP process is not that proof.

- **LLM:** open `add-llm-profile`, fill `profile-name-input`, select a real provider
  and model (or `sdk-section-all-toggle` / `llm-custom-model-input` / `base-url-input`).
  Fill `llm-api-key-input`, submit `save-profile-btn`, then reload and inspect the
  named `profile-row`. Exercise scoped `profile-menu-trigger` edit/default/delete
  actions; verify persisted data and the actual model used in a bounded conversation.
- **Connections:** open `add-provider-connection`; inspect `provider-connection-modal`
  with safe dummy credentials at both viewports. For authorized persistence checks,
  save a dedicated connection, select it from a profile, rotate only its test key,
  and verify the dependent profile still works without exposing values.
- **Agent/ACP:** select a supported installed agent, configure its real command in
  `agent-command-input` when offered, and save with `save-agent-profile-btn`.
  Inspect missing-auth guidance, then prove a real response when credentials exist.
- **Switch:** use `chat-input-llm-profile` and the conversation's model-switch action;
  verify effective model metadata after a real response and after reopening.

## Gotchas

Saving a profile may make a validation completion call: do not click Save with a
real provider merely to take screenshots. A green key-present indicator is not
credential validity. Preserve native password fields for styling evidence. Never
rotate a production key or copy another user's profile store into QA state.
