import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";

function CondenserSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["condenser"]}
      testId="condenser-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default CondenserSettingsScreen;
