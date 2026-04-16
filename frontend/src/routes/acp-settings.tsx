import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

/**
 * Settings page for the ACP (Agent Client Protocol) agent variant.
 *
 * Shows the ``agent_kind`` discriminator (in the schema's "general" section)
 * plus the "acp" section fields (acp_command, acp_model, acp_args, …).
 * The section-variant filter in ``getVisibleSettingsSections`` automatically
 * hides sections that don't apply to the selected agent_kind.
 */
function AcpSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["general", "acp"]}
      testId="acp-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default AcpSettingsScreen;
