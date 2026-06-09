import { useTranslation } from "react-i18next";
import { AcpSecretField } from "#/components/features/settings/acp-secret-field";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import type { AcpCredentialForm } from "#/hooks/use-acp-credential-form";

/**
 * Settings → Agent credentials section: renders API-key and base-URL fields for
 * built-in ACP providers (Claude Code, Codex). The form state and the save are
 * owned by the parent (Settings → Agent) so the page has a single Save button
 * for both agent settings and credentials. Returns null for providers with no
 * credential fields (Gemini CLI, custom preset).
 */
export function AcpCredentialsSection({ form }: { form: AcpCredentialForm }) {
  const { t } = useTranslation();
  const { fields, values, setValue, secretExists } = form;

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
    </div>
  );
}
