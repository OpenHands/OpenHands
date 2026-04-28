import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

function CondenserSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["condenser"]}
      testId="condenser-settings-screen"
    />
  );
}

const orgDefaultsRedirectGuard = requireOrgDefaultsRedirect(
  "/settings/org-defaults/condenser",
);
const condenserPermissionGuard = createPermissionGuard("view_llm_settings");

export const clientLoader = async (args: { request: Request }) => {
  const blocked = await orgDefaultsRedirectGuard(args);
  if (blocked) return blocked;
  return condenserPermissionGuard(args);
};

export default CondenserSettingsScreen;
