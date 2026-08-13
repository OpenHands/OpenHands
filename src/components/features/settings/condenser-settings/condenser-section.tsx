import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

/**
 * Condenser settings, rendered from the agent schema's `condenser` section.
 * Section-owned save is handled by {@link SdkSectionPage} (it persists only the
 * fields it renders). Unchanged in behaviour from the former inline page body —
 * it is just registered as a section now so the Condenser page uses the same
 * host as every other settings page.
 */
export function CondenserSection() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "agent_settings", sectionKeys: ["condenser"] },
      ]}
      testId="condenser-settings-screen"
    />
  );
}
