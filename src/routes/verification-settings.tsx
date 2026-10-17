import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { ActiveProfileSectionPage } from "#/components/features/settings/sdk-settings/active-profile-section-page";
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

const VERIFICATION_SOURCES = [
  {
    settingsSource: "conversation_settings" as const,
    sectionKeys: ["verification"],
  },
  {
    settingsSource: "agent_settings" as const,
    sectionKeys: ["verification"],
    excludeKeys: CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS,
  },
];

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
      settingsSources={VERIFICATION_SOURCES}
      header={renderTopContent ? () => renderTopContent() : undefined}
      testId={testId}
    />
  );
}

/**
 * The critic's behavior belongs to the active agent profile; its API key,
 * confirmation mode and security analyzer stay global.
 */
function VerificationSettingsRoute() {
  return (
    <ActiveProfileSectionPage
      settingsSources={VERIFICATION_SOURCES}
      section="verification"
      globalKeys={["critic_api_key"]}
      testId="verification-settings-screen"
    />
  );
}

export default VerificationSettingsRoute;
