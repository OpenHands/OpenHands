import { useTranslation } from "react-i18next";
import { AcpConflictWarnings } from "#/components/features/settings/acp-conflict-warnings";
import { AcpAuthStatusBanner } from "#/components/features/settings/acp-auth-status-banner";
import { AcpSecretField } from "#/components/features/settings/acp-secret-field";
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
  const { fields, values, setValue, secretExists, conflicts } = form;
  const { status: authStatus, isChecking } = useAcpAuthStatus(providerKey);
  const providerName = getAcpProviderDisplayName(providerKey) ?? providerKey;

  if (fields.length === 0) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <Typography.Text className="text-sm font-medium text-white">
          {t(I18nKey.SETTINGS$ACP_CREDENTIALS_TITLE)}
        </Typography.Text>
        <Typography.Text className="text-xs text-[#717888]">
          {t(I18nKey.SETTINGS$ACP_CREDENTIALS_DESCRIPTION)}
        </Typography.Text>
      </div>

      <AcpAuthStatusBanner
        status={authStatus}
        isChecking={isChecking}
        providerName={providerName}
        testIdPrefix="settings-acp-auth"
      />

      <div className="flex flex-col gap-5">
        {fields.map((field) => (
          <AcpSecretField
            key={field.name}
            field={field}
            value={values[field.name] ?? ""}
            onChange={(value) => setValue(field.name, value)}
            alreadySet={secretExists(field.name)}
            testId={`settings-acp-secret-${field.name}`}
            showOptionalTag
          />
        ))}
      </div>

      <AcpConflictWarnings conflicts={conflicts} />
    </div>
  );
}
