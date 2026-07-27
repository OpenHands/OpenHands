import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function MemorySettingsScreen() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "agent_settings", sectionKeys: ["agent_context"] },
      ]}
      testId="memory-settings-screen"
    />
  );
}

export default MemorySettingsScreen;
