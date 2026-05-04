import { useTranslation } from "react-i18next";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { PROFILE_NAME_PATTERN } from "#/utils/derive-profile-name";

interface ProfileNameInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  isDisabled?: boolean;
  /** Render label as "Name (Optional)" when this field isn't required. */
  isOptional?: boolean;
  testId?: string;
  ruleTestId?: string;
}

export function ProfileNameInput({
  value,
  onChange,
  placeholder,
  isDisabled,
  isOptional,
  testId,
  ruleTestId,
}: ProfileNameInputProps) {
  const { t } = useTranslation();
  const trimmed = value.trim();
  const isInvalid = trimmed.length > 0 && !PROFILE_NAME_PATTERN.test(trimmed);
  const label = isOptional
    ? `${t(I18nKey.SETTINGS$NAME)} (${t(I18nKey.COMMON$OPTIONAL)})`
    : t(I18nKey.SETTINGS$NAME);

  return (
    <div className="flex flex-col gap-1">
      <SettingsInput
        testId={testId}
        label={label}
        type="text"
        className="w-full"
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        isDisabled={isDisabled}
      />
      <Typography.Paragraph
        testId={ruleTestId}
        className={`text-xs ${isInvalid ? "text-red-400" : "text-gray-400"}`}
      >
        {t(I18nKey.SETTINGS$PROFILE_NAME_RULE)}
      </Typography.Paragraph>
    </div>
  );
}
