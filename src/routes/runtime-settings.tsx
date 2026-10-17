import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function RuntimeSettingsScreen() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "conversation_settings", sectionKeys: ["runtime"] },
      ]}
      testId="runtime-settings-screen"
    />
  );
}

export default RuntimeSettingsScreen;
