import { Navigate } from "react-router";
import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { useActiveBackend } from "#/contexts/active-backend-context";

function CondenserSettingsScreen() {
  const { backend } = useActiveBackend();
  // Locally each agent profile has its own condenser.
  if (backend.kind !== "cloud") {
    return <Navigate to="/settings/agents" replace />;
  }
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
