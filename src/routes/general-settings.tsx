import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SettingsScope } from "#/types/settings";

/**
 * Renders the "general" conversation-settings section. Today it carries
 * exactly one schema field: `max_iterations` — the session-wide execution
 * budget (maximum number of agent steps) applied when conversations start.
 * It is rendered through the schema-driven `SdkSectionPage`, so the input is
 * always derived from the live agent-server schema (`ge=1`, no artificial
 * upper cap) rather than a hand-maintained field definition.
 *
 * This is a session default: the per-command `/goal --max <n>` flag (see
 * `use-goal-interceptor`) still overrides it for individual goal runs.
 */
export function GeneralSettingsScreen({
  scope = "personal",
  renderTopContent,
  testId = "general-settings-screen",
}: {
  scope?: SettingsScope;
  renderTopContent?: () => React.ReactNode;
  testId?: string;
}) {
  return (
    <SdkSectionPage
      scope={scope}
      settingsSources={[
        {
          settingsSource: "conversation_settings",
          sectionKeys: ["general"],
        },
      ]}
      header={renderTopContent ? () => renderTopContent() : undefined}
      testId={testId}
    />
  );
}

export default GeneralSettingsScreen;
