import { OrgDefaultsBanner } from "#/components/features/settings/org-defaults-banner";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { VerificationSettingsScreen } from "./verification-settings";

const renderOrgDefaultsBanner = () => <OrgDefaultsBanner />;

function OrgDefaultVerificationSettingsScreen() {
  return (
    <VerificationSettingsScreen
      scope="org"
      renderTopContent={renderOrgDefaultsBanner}
      testId="org-default-verification-settings-screen"
    />
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default OrgDefaultVerificationSettingsScreen;
