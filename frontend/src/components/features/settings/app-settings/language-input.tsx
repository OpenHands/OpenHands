import type { Key } from "react";
import { useTranslation } from "react-i18next";
import { AvailableLanguages } from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { SettingsDropdownInput } from "../settings-dropdown-input";

interface LanguageInputProps {
  name: string;
  onChange: (value: string) => void;
  defaultKey: string;
}

export function LanguageInput({
  defaultKey,
  onChange,
  name,
}: LanguageInputProps) {
  const { t } = useTranslation();
  const handleSelectionChange = (key: Key | null) => {
    const language = AvailableLanguages.find((l) => l.value === key);
    if (language) {
      onChange(language.label);
    }
  };

  return (
    <SettingsDropdownInput
      testId={name}
      name={name}
      onSelectionChange={handleSelectionChange}
      label={t(I18nKey.SETTINGS$LANGUAGE)}
      items={AvailableLanguages.map((l) => ({
        key: l.value,
        label: l.label,
      }))}
      defaultSelectedKey={defaultKey}
      isClearable={false}
      wrapperClassName="w-full max-w-[680px]"
    />
  );
}
