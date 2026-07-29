import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SettingsScope } from "#/types/settings";

// Defensive de-dup: agent_settings.verification still carries
// `confirmation_mode` and `security_analyzer` for back-compat, but the SDK
// deprecated them and moved the canonical copies to ConversationSettings.
// Render only the conversation-source versions so these fields don't show
// up twice on the page.
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

export default VerificationSettingsScreen;
