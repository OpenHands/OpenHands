import React from "react";
import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SettingsScope } from "#/types/settings";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

function VerificationSettingsHeader({
  renderTopContent,
}: {
  renderTopContent?: () => React.ReactNode;
}) {
  return <div className="flex flex-col gap-6">{renderTopContent?.()}</div>;
}

export function VerificationSettingsScreen({
  scope = "personal",
  renderTopContent,
  testId = "verification-settings-screen",
}: {
  scope?: SettingsScope;
  renderTopContent?: () => React.ReactNode;
  testId?: string;
}) {
  const buildHeader = React.useCallback(
    () => <VerificationSettingsHeader renderTopContent={renderTopContent} />,
    [renderTopContent],
  );

  return (
    <SdkSectionPage
      scope={scope}
      settingsSource="conversation_settings"
      sectionKeys={["verification"]}
      header={buildHeader}
      testId={testId}
    />
  );
}

const orgDefaultsRedirectGuard = requireOrgDefaultsRedirect(
  "/settings/org-defaults/verification",
);
const verificationPermissionGuard = createPermissionGuard("view_llm_settings");

export const clientLoader = async (args: { request: Request }) => {
  const blocked = await orgDefaultsRedirectGuard(args);
  if (blocked) return blocked;
  return verificationPermissionGuard(args);
};

export default VerificationSettingsScreen;
