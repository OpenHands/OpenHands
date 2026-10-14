import { ActiveProfileSectionPage } from "#/components/features/settings/sdk-settings/active-profile-section-page";

const CONDENSER_SOURCES = [
  { settingsSource: "agent_settings" as const, sectionKeys: ["condenser"] },
];

function CondenserSettingsScreen() {
  return (
    <ActiveProfileSectionPage
      settingsSources={CONDENSER_SOURCES}
      section="condenser"
      testId="condenser-settings-screen"
    />
  );
}

export default CondenserSettingsScreen;
