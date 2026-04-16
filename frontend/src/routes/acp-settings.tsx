import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

/**
 * Settings page for the ACP (Agent Client Protocol) agent variant.
 *
 * Renders the "acp" section plus the ACP-variant LLM section (used
 * purely for cost/token attribution — the real completions happen in
 * the ACP subprocess). Sections and fields tagged ``variant: "llm"``
 * in the schema are hidden, so the CodeActAgent / tools / MCP fields
 * do not appear here.
 *
 * Saving always commits ``agent_kind: "acp"`` so the server-side
 * discriminator routes the agent through the ACP conversation path.
 */
function AcpSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["acp", "llm"]}
      variant="acp"
      buildPayload={(basePayload) => {
        // basePayload is a nested dict built from the dirty fields.
        // Inject the discriminator so the backend validates this as
        // an ACPAgentSettings payload.
        const agentSettings = {
          ...(basePayload as Record<string, unknown>),
          agent_kind: "acp",
        };
        return { agent_settings: agentSettings };
      }}
      testId="acp-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default AcpSettingsScreen;
