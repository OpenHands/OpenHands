import React, { forwardRef } from "react";
import { useTranslation } from "react-i18next";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { isProfileNameValid } from "#/utils/derive-profile-name";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ProfileNameInputProps {
  testId?: string;
  ruleTestId?: string;
  value: string;
  onChange: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  placeholder?: string;
  isDisabled?: boolean;
  /** Render label as "Name (Optional)" when this field isn't required. */
  isOptional?: boolean;
  /** When true, empty values will show red validation styling (required field behavior). */
  isRequired?: boolean;
}

export const ProfileNameInput = forwardRef<
  HTMLInputElement,
  ProfileNameInputProps
>(function ProfileNameInput(
  {
    testId,
    ruleTestId,
    value,
    onChange,
    onKeyDown,
    placeholder,
    isDisabled,
    isOptional,
    isRequired = false,
  },
  ref,
) {
  const { t } = useTranslation("openhands");
  const isValid = isProfileNameValid(value, { isRequired });
  const label = isOptional
    ? `${t(I18nKey.SETTINGS$PROFILE_NAME_LABEL)} (${t(I18nKey.COMMON$OPTIONAL)})`
    : t(I18nKey.SETTINGS$PROFILE_NAME_LABEL);

  // Generate a stable ID for the rule element to link with aria-describedby
  const ruleId = React.useId();
  const describedById = ruleTestId ?? `${ruleId}-rule`;

  return (
    <div className="flex flex-col gap-2">
      <SettingsInput
        ref={ref}
        testId={testId}
        label={label}
        type="text"
        className="w-full"
        value={value}
        placeholder={
          placeholder ?? t(I18nKey.SETTINGS$PROFILE_NAME_PLACEHOLDER)
        }
        onChange={onChange}
        onKeyDown={onKeyDown}
        isDisabled={isDisabled}
        ariaDescribedBy={describedById}
        ariaInvalid={!isValid}
      />
      <p
        id={describedById}
        data-testid={ruleTestId}
        className={cn(
          "text-xs",
          isValid ? "text-[var(--oh-muted)]" : "text-red-400",
        )}
      >
        {t(I18nKey.SETTINGS$PROFILE_NAME_RULE)}
      </p>
    </div>
  );
});
