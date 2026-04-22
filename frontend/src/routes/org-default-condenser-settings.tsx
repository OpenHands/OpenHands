import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { OrgDefaultsBanner } from "#/components/features/settings/org-defaults-banner";
import { createPermissionGuard } from "#/utils/org/permission-guard";

const renderOrgDefaultsBanner = () => <OrgDefaultsBanner />;

function OrgDefaultCondenserSettingsScreen() {
  return (
    <SdkSectionPage
      scope="org"
      sectionKeys={["condenser"]}
      header={renderOrgDefaultsBanner}
      testId="org-default-condenser-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default OrgDefaultCondenserSettingsScreen;
