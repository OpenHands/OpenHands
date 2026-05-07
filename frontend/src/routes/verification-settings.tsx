import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SettingsScope } from "#/types/settings";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

// Defensive de-dup: agent_settings.verification still carries
// `confirmation_mode` and `security_analyzer` for back-compat, but the SDK
// deprecated them in 1.17.0 (see VerificationSettings field validators in
// openhands-sdk) and moved the canonical copies to ConversationSettings.
// Render only the conversation-source versions so these fields don't show
// up twice on the page.
//
// TODO: drop this set once openhands-sdk 1.22.0 lands and the deprecated
// VerificationSettings.{confirmation_mode,security_analyzer} fields are
// fully removed.
const CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS = new Set([
  "verification.confirmation_mode",
  "verification.security_analyzer",
]);

export function VerificationSettingsScreen({
  scope = "personal",
  renderTopContent,
  testId = "verification-settings-screen",
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
          sectionKeys: ["verification"],
        },
        {
          settingsSource: "agent_settings",
          sectionKeys: ["verification"],
          excludeKeys: CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS,
        },
      ]}
      header={renderTopContent ? () => renderTopContent() : undefined}
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
