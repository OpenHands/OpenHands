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
  const { t } = useTranslation("openhands");

  return (
    <SettingsDropdownInput
      testId={name}
      name={name}
      onInputChange={onChange}
      label={t(I18nKey.SETTINGS$LANGUAGE)}
      items={AvailableLanguages.map((l) => ({
        key: l.value,
        label: l.label,
      }))}
      defaultSelectedKey={defaultKey}
      isClearable={false}
      wrapperClassName="w-full min-w-0"
    />
  );
}
