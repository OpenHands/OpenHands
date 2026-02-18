import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { KeyStatusIcon } from "../key-status-icon";

interface NeonApiKeyInputProps {
  onChange: (value: string) => void;
  isNeonKeySet: boolean;
}

export function NeonApiKeyInput({
  onChange,
  isNeonKeySet,
}: NeonApiKeyInputProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6">
      <SettingsInput
        testId="neon-api-key-input"
        name="neon-api-key-input"
        onChange={onChange}
        label={t(I18nKey.NEON$API_KEY_LABEL)}
        type="password"
        className="w-full max-w-[680px]"
        placeholder={isNeonKeySet ? "<hidden>" : ""}
        startContent={
          isNeonKeySet && (
            <KeyStatusIcon
              testId="neon-set-key-indicator"
              isSet={isNeonKeySet}
            />
          )
        }
      />
    </div>
  );
}
