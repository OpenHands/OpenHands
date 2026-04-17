import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

/**
 * Settings page for ACP *subprocess launch* configuration + the
 * provider credentials the subprocess needs to authenticate:
 *
 * - ACP section — basic: ``acp_server`` (Claude Code / Codex /
 *   Gemini CLI / Custom). Advanced/all: ``acp_command``, ``acp_args``,
 *   ``acp_env`` for overriding the default subprocess command.
 * - LLM section — basic: ``llm.api_key`` + ``llm.base_url`` (the
 *   backend translates these into the right provider env vars for the
 *   chosen ``acp_server`` — see ``LiveStatusAppConversationService
 *   ._acp_provider_env``). ``llm.model`` is excluded because the
 *   authoritative model for ACP is ``acp_model`` on the ACP Model
 *   page.
 *
 * The ``acp_model``, ``acp_session_mode``, and ``acp_prompt_timeout``
 * fields are excluded here — they belong on the companion "ACP Model"
 * page. Saving commits ``agent_kind: "acp"`` so landing here while in
 * OpenHands mode and saving switches the agent type over.
 */
const EXCLUDED_KEYS = new Set<string>([
  // Lives on the ACP Model page.
  "acp_model",
  "acp_session_mode",
  "acp_prompt_timeout",
  // ``acp_model`` is the authoritative model identifier for ACP; the
  // ``llm.model`` field on ``ACPAgentSettings`` is only used for cost
  // attribution inside OpenHands and would confuse the user if shown.
  "llm.model",
]);

function AcpServerSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["acp", "llm"]}
      variant="acp"
      excludeKeys={EXCLUDED_KEYS}
      buildPayload={(basePayload) => {
        const agentSettings = {
          ...(basePayload as Record<string, unknown>),
          agent_kind: "acp",
        };
        return { agent_settings: agentSettings };
      }}
      testId="acp-server-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default AcpServerSettingsScreen;
