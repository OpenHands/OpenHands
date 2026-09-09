import { GraphIndexerControls } from "#/components/features/graph/graph-page";
import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";

function AgentContextSettingsScreen() {
  return (
    <div className="flex flex-col gap-8">
      <SdkSectionPage
        settingsSources={[
          { settingsSource: "agent_settings", sectionKeys: ["agent_context"] },
        ]}
        testId="agent-context-settings-screen"
      />
      <GraphIndexerControls />
    </div>
  );
}

export default AgentContextSettingsScreen;
