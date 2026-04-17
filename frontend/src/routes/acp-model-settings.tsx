import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

/**
 * Settings page for ACP *model* configuration — what the ACP
 * subprocess actually runs once launched:
 *
 * - Basic view: ``acp_model`` (e.g. ``claude-opus-4-6``).
 * - Advanced/all: ``acp_session_mode``, ``acp_prompt_timeout``.
 *
 * Subprocess launch details (``acp_server`` + custom command/args/env)
 * live on the companion "ACP Server" page. Saving commits
 * ``agent_kind: "acp"``.
 */
const ACP_SERVER_FIELDS = new Set<string>([
  "acp_server",
  "acp_command",
  "acp_args",
  "acp_env",
]);

function AcpModelSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["acp"]}
      variant="acp"
      excludeKeys={ACP_SERVER_FIELDS}
      buildPayload={(basePayload) => {
        const agentSettings = {
          ...(basePayload as Record<string, unknown>),
          agent_kind: "acp",
        };
        return { agent_settings: agentSettings };
      }}
      testId="acp-model-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default AcpModelSettingsScreen;
