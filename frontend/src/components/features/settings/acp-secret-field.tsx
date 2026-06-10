import { useTranslation } from "react-i18next";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { OptionalTag } from "#/components/features/settings/optional-tag";
import { I18nKey } from "#/i18n/declaration";
import type { ACPProviderSecretField } from "#/constants/acp-provider-secrets";

interface AcpSecretFieldProps {
  field: ACPProviderSecretField;
  value: string;
  onChange: (value: string) => void;
  alreadySet: boolean;
  testId: string;
  showOptionalTag?: boolean;
}

/**
 * A single ACP credential field — either a multiline textarea for file-content
 * blobs or a masked/plain SettingsInput for keys and base URLs.
 */
export function AcpSecretField({
  field,
  value,
  onChange,
  alreadySet,
  testId,
  showOptionalTag,
}: AcpSecretFieldProps) {
  const { t } = useTranslation();
  const placeholder = alreadySet
    ? t(I18nKey.SETTINGS$ACP_SECRET_ALREADY_SET)
    : "";

  return (
    <div className="flex flex-col gap-1.5">
      {field.multiline ? (
        <label className="flex flex-col gap-2.5">
          <span className="flex items-center gap-2">
            <span className="text-sm font-mono text-white">{field.name}</span>
            {showOptionalTag && <OptionalTag />}
          </span>
          <textarea
            data-testid={testId}
            name={field.name}
            rows={4}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            value={value}
            placeholder={placeholder}
            onChange={(e) => onChange(e.target.value)}
            className="bg-tertiary border border-[#717888] rounded-sm p-2 text-xs font-mono text-white placeholder:italic placeholder:text-[#717888] min-h-[80px] resize-y focus:outline-none focus:border-white"
          />
        </label>
      ) : (
        <SettingsInput
          testId={testId}
          name={field.name}
          label={field.name}
          labelClassName="font-mono"
          type={field.secret ? "password" : "text"}
          value={value}
          onChange={onChange}
          showOptionalTag={showOptionalTag}
          placeholder={placeholder}
          autoComplete={field.secret ? "new-password" : "off"}
        />
      )}
      <span className="text-xs text-[#717888]">
        {t(field.hint_key, field.hint_values)}
      </span>
    </div>
  );
}
