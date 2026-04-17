import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

/**
 * Settings page for ACP *subprocess launch* configuration:
 *
 * - Basic view: ``acp_server`` (Claude Code / Codex / Gemini CLI / Custom).
 * - Advanced/all: ``acp_command``, ``acp_args``, ``acp_env`` for
 *   overriding the default subprocess command.
 *
 * The ``acp_model``, ``acp_session_mode``, and ``acp_prompt_timeout``
 * fields are excluded here — they belong on the companion "ACP Model"
 * page. Saving commits ``agent_kind: "acp"`` so landing here while in
 * OpenHands mode and saving switches the agent type over.
 */
const ACP_MODEL_FIELDS = new Set<string>([
  "acp_model",
  "acp_session_mode",
  "acp_prompt_timeout",
]);

function AcpServerSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["acp"]}
      variant="acp"
      excludeKeys={ACP_MODEL_FIELDS}
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
