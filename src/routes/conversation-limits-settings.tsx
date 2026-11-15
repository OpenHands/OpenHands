import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function ConversationLimitsSettingsScreen() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "conversation_settings", sectionKeys: ["general"] },
      ]}
      testId="conversation-limits-settings-screen"
    />
  );
}

export default ConversationLimitsSettingsScreen;
