import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { OptionalTag } from "#/components/features/settings/optional-tag";
import { I18nKey } from "#/i18n/declaration";
import { type ACPProviderSecretField } from "#/constants/acp-providers";

interface AcpSecretFieldProps {
  field: ACPProviderSecretField;
  value: string;
  onChange: (value: string) => void;
  alreadySet: boolean;
  testId: string;
  showOptionalTag?: boolean;
}

/**
 * Renders a single ACP credential field — a multiline textarea for file-content
 * blobs (Codex auth.json, Gemini SA JSON) or a masked/plain {@link SettingsInput}
 * for everything else — plus its hint text. Used by both the onboarding
 * credentials step and the Settings → Agent credentials section.
 */
export function AcpSecretField({
  field,
  value,
  onChange,
  alreadySet,
  testId,
  showOptionalTag,
}: AcpSecretFieldProps) {
  const { t } = useTranslation("openhands");
  const placeholder = alreadySet
    ? t(I18nKey.ONBOARDING$ACP_SECRET_ALREADY_SET)
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
            className={cn(
              formControlMultilineFieldClassName,
              "font-mono text-xs",
            )}
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
        />
      )}
      <span className="text-xs text-[var(--oh-muted)]">
        {t(field.hint_key, field.hint_values)}
      </span>
    </div>
  );
}
