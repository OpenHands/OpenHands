import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function CondenserSettingsScreen() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "agent_settings", sectionKeys: ["condenser"] },
      ]}
      testId="condenser-settings-screen"
    />
  );
}

export default CondenserSettingsScreen;
