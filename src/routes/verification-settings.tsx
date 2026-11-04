import React from "react";
import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SdkSectionPage } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { profileBehaviorFieldKeys } from "#/components/features/settings/agent-profiles/profile-behavior-form";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
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

// The critic API key depends on `critic_enabled`, which each profile sets.
const SHOW_CRITIC_API_KEY = { "verification.critic_enabled": true };

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

/** Locally each agent profile owns the critic's behavior; its API key stays here. */
function GlobalVerificationSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const settingsSources = React.useMemo(
    () =>
      schema
        ? [
            {
              settingsSource: "agent_settings" as const,
              sectionKeys: ["verification"],
              excludeKeys: new Set([
                ...CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS,
                ...profileBehaviorFieldKeys(schema, "verification"),
              ]),
            },
            {
              settingsSource: "conversation_settings" as const,
              sectionKeys: ["verification"],
            },
          ]
        : null,
    [schema],
  );
  if (!settingsSources) return null;
  return (
    <div className="flex flex-col gap-4">
      <Typography.Text
        testId="verification-critic-per-profile-note"
        className="text-xs text-tertiary-alt"
      >
        {t(I18nKey.SETTINGS$VERIFICATION_CRITIC_PER_PROFILE)}
      </Typography.Text>
      <SdkSectionPage
        settingsSources={settingsSources}
        initialValueOverrides={SHOW_CRITIC_API_KEY}
        markInitialOverridesDirty={false}
        testId="verification-settings-screen"
      />
    </div>
  );
}

function VerificationSettingsRoute() {
  const { backend } = useActiveBackend();
  return backend.kind === "cloud" ? (
    <VerificationSettingsScreen />
  ) : (
    <GlobalVerificationSettingsScreen />
  );
}

export default VerificationSettingsRoute;
