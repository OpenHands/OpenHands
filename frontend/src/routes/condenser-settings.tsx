import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requirePersonalWorkspaceLoader } from "#/utils/org/personal-workspace-guard";

function CondenserSettingsScreen() {
  return (
    <SdkSectionPage
      sectionKeys={["condenser"]}
      testId="condenser-settings-screen"
    />
  );
}

const personalWorkspaceGuard = requirePersonalWorkspaceLoader(
  "/settings/org-defaults/condenser",
);
const condenserPermissionGuard = createPermissionGuard("view_llm_settings");

export const clientLoader = async (args: { request: Request }) => {
  const blocked = await personalWorkspaceGuard(args);
  if (blocked) return blocked;
  return condenserPermissionGuard(args);
};

export default CondenserSettingsScreen;
