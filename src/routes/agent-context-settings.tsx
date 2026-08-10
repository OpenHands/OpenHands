import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function AgentContextSettingsScreen() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "agent_settings", sectionKeys: ["agent_context"] },
      ]}
      testId="agent-context-settings-screen"
    />
  );
}

export default AgentContextSettingsScreen;
