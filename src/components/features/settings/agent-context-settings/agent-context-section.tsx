import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

/**
 * Agent-context settings, rendered from the agent schema's `agent_context`
 * section (today only persistent memory). Section-owned save is handled by
 * {@link SdkSectionPage}. Unchanged in behaviour from the former inline page
 * body — it is just registered as a section now so the Agent Context page uses
 * the same host as every other settings page.
 */
export function AgentContextSection() {
  return (
    <SdkSectionPage
      settingsSources={[
        { settingsSource: "agent_settings", sectionKeys: ["agent_context"] },
      ]}
      testId="agent-context-settings-screen"
    />
  );
}
