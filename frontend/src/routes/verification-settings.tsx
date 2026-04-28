import React from "react";
import { useTranslation } from "react-i18next";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import {
  SdkSectionHeaderProps,
  SdkSectionPage,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { SettingsScope } from "#/types/settings";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

// These fields are rendered manually in the header, so exclude them from
// schema-driven rendering.  Agent-settings keys are dot-prefixed with the
// section name (e.g. "verification.confirmation_mode").
const VERIFICATION_SCHEMA_EXCLUDE_KEYS = new Set([
  "verification.confirmation_mode",
  "verification.security_analyzer",
]);

// Promote iterative refinement to Basic view so it's discoverable alongside "Enable Critic"
const VERIFICATION_PROMINENCE_OVERRIDES: Record<
  string,
  "critical" | "major" | "minor"
> = {
  "verification.enable_iterative_refinement": "critical",
};

function VerificationSettingsHeader({
  confirmationMode,
  securityAnalyzer,
  isConversationSettingsDisabled,
  onConfirmationModeChange,
  onSecurityAnalyzerChange,
  renderTopContent,
}: {
  confirmationMode: boolean;
  securityAnalyzer: string | null;
  isConversationSettingsDisabled: boolean;
  onConfirmationModeChange: (value: boolean) => void;
  onSecurityAnalyzerChange: (value: string | null) => void;
  renderTopContent?: () => React.ReactNode;
}) {
  const { t } = useTranslation();

  const securityAnalyzerItems = React.useMemo(
    () => [
      {
        key: "llm",
        label: t(I18nKey.SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT),
      },
      {
        key: "none",
        label: t(I18nKey.SETTINGS$SECURITY_ANALYZER_NONE),
      },
    ],
    [t],
  );

  const showSecurityAnalyzer = confirmationMode;

  return (
    <div className="flex flex-col gap-6">
      {renderTopContent?.()}

      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-1.5">
          <SettingsSwitch
            testId="confirmation-mode-toggle"
            isToggled={confirmationMode}
            onToggle={onConfirmationModeChange}
            isDisabled={isConversationSettingsDisabled}
          >
            {t(I18nKey.SETTINGS_FORM$ENABLE_CONFIRMATION_MODE_LABEL)}
          </SettingsSwitch>
          <p className="text-tertiary-alt text-xs leading-5">
            {t(I18nKey.SETTINGS$CONFIRMATION_MODE_TOOLTIP)}
          </p>
        </div>

        {showSecurityAnalyzer ? (
          <div className="flex flex-col gap-1.5">
            <SettingsDropdownInput
              testId="security-analyzer-input"
              name="security_analyzer"
              label={t(I18nKey.SETTINGS_FORM$SECURITY_ANALYZER_LABEL)}
              items={securityAnalyzerItems}
              selectedKey={securityAnalyzer ?? undefined}
              placeholder={t(I18nKey.SETTINGS$SECURITY_ANALYZER_PLACEHOLDER)}
              isDisabled={isConversationSettingsDisabled}
              onSelectionChange={(key) =>
                onSecurityAnalyzerChange(key ? String(key) : null)
              }
            />
            <p className="text-tertiary-alt text-xs leading-5 max-w-[680px] ">
              {t(I18nKey.SETTINGS$SECURITY_ANALYZER_DESCRIPTION)}
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
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
  const { data: settings } = useSettings(scope);
  const [confirmationMode, setConfirmationMode] = React.useState(
    DEFAULT_SETTINGS.confirmation_mode,
  );
  const [securityAnalyzer, setSecurityAnalyzer] = React.useState<string | null>(
    DEFAULT_SETTINGS.security_analyzer,
  );
  const [confirmationModeDirty, setConfirmationModeDirty] =
    React.useState(false);
  const [securityAnalyzerDirty, setSecurityAnalyzerDirty] =
    React.useState(false);

  React.useEffect(() => {
    setConfirmationMode(
      settings?.confirmation_mode ?? DEFAULT_SETTINGS.confirmation_mode,
    );
    setSecurityAnalyzer(
      settings?.security_analyzer ?? DEFAULT_SETTINGS.security_analyzer,
    );
    setConfirmationModeDirty(false);
    setSecurityAnalyzerDirty(false);
  }, [settings?.confirmation_mode, settings?.security_analyzer]);

  const buildHeader = React.useCallback(
    ({ isDisabled }: SdkSectionHeaderProps) => (
      <VerificationSettingsHeader
        confirmationMode={confirmationMode}
        securityAnalyzer={securityAnalyzer}
        isConversationSettingsDisabled={isDisabled}
        onConfirmationModeChange={(value) => {
          setConfirmationMode(value);
          setConfirmationModeDirty(true);
        }}
        onSecurityAnalyzerChange={(value) => {
          setSecurityAnalyzer(value);
          setSecurityAnalyzerDirty(true);
        }}
        renderTopContent={renderTopContent}
      />
    ),
    [confirmationMode, renderTopContent, securityAnalyzer],
  );

  const buildPayload = React.useCallback(
    (basePayload: Record<string, unknown>) => {
      // Critic fields go into agent_settings_diff
      const result: Record<string, unknown> = {};
      if (Object.keys(basePayload).length > 0) {
        result.agent_settings_diff = basePayload;
      }

      // Confirmation mode and security analyzer go into conversation_settings_diff
      const conversationDiff: Record<string, unknown> = {};
      if (confirmationModeDirty) {
        conversationDiff.confirmation_mode = confirmationMode;
      }
      if (
        securityAnalyzerDirty ||
        (confirmationMode && settings?.security_analyzer !== securityAnalyzer)
      ) {
        conversationDiff.security_analyzer = securityAnalyzer;
      }
      if (Object.keys(conversationDiff).length > 0) {
        result.conversation_settings_diff = conversationDiff;
      }

      return result;
    },
    [
      confirmationMode,
      confirmationModeDirty,
      securityAnalyzer,
      securityAnalyzerDirty,
      settings?.security_analyzer,
    ],
  );

  return (
    <SdkSectionPage
      scope={scope}
      settingsSource="agent_settings"
      sectionKeys={["verification"]}
      excludeKeys={VERIFICATION_SCHEMA_EXCLUDE_KEYS}
      prominenceOverrides={VERIFICATION_PROMINENCE_OVERRIDES}
      header={buildHeader}
      extraDirty={confirmationModeDirty || securityAnalyzerDirty}
      buildPayload={buildPayload}
      onSaveSuccess={() => {
        setConfirmationModeDirty(false);
        setSecurityAnalyzerDirty(false);
      }}
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
