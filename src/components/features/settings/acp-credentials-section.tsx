import { useTranslation } from "react-i18next";
import { AcpConflictWarnings } from "#/components/features/settings/acp-conflict-warnings";
import { AcpAuthStatusBanner } from "#/components/features/settings/acp-auth-status-banner";
import { AcpCredentialFields } from "#/components/features/settings/acp-credential-fields";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import { getAcpProviderDisplayName } from "#/constants/acp-providers";
import type { AcpCredentialForm } from "#/hooks/use-acp-credential-form";

/**
 * Settings → Agent credentials section for a built-in ACP provider: renders the
 * same fields the onboarding step collects (and the same "already signed in"
 * auth banner), so credentials can be added or rotated after onboarding. The
 * form state and the save are owned by the parent (Settings → Agent) so the
 * page has a single Save button for both agent settings and credentials.
 * Renders nothing for providers without credential fields.
 */
export function AcpCredentialsSection({
  form,
  providerKey,
}: {
  form: AcpCredentialForm;
  providerKey: string;
}) {
  const { t } = useTranslation("openhands");
  const { fields, conflicts, credentialsConfigured } = form;
  const { status: authStatus, isChecking } = useAcpAuthStatus(providerKey);
  const providerName = getAcpProviderDisplayName(providerKey) ?? providerKey;

  if (fields.length === 0) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <Typography.Text className="text-sm font-medium text-contrast">
          {t(I18nKey.SETTINGS$ACP_CREDENTIALS_TITLE)}
        </Typography.Text>
        <Typography.Text className="text-xs text-[#717888]">
          {t(I18nKey.SETTINGS$ACP_CREDENTIALS_DESCRIPTION)}
        </Typography.Text>
      </div>

      <AcpAuthStatusBanner
        status={authStatus}
        isChecking={isChecking}
        credentialsConfigured={credentialsConfigured}
        providerName={providerName}
        testIdPrefix="settings-acp-auth"
      />

      <AcpCredentialFields
        form={form}
        isAuthenticated={authStatus === "authenticated"}
        testIdPrefix="settings-acp"
      />

      <AcpConflictWarnings conflicts={conflicts} />
    </div>
  );
}
