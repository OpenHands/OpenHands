import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { KeyStatusIcon } from "../key-status-icon";
import { BrandButton } from "../brand-button";
import { cn } from "#/utils/utils";

interface ForgejoTokenInputProps {
  onChange: (value: string) => void;
  onForgejoHostChange: (value: string) => void;
  isForgejoTokenSet: boolean;
  name: string;
  forgejoHostSet: string | null | undefined;
  className?: string;
  onClear?: () => void;
}

export function ForgejoTokenInput({
  onChange,
  onForgejoHostChange,
  isForgejoTokenSet,
  name,
  forgejoHostSet,
  className,
  onClear,
}: ForgejoTokenInputProps) {
  const { t } = useTranslation();

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex items-end gap-2">
        <SettingsInput
          testId={name}
          name={name}
          onChange={onChange}
          label={t(I18nKey.FORGEJO$TOKEN_LABEL)}
          type="password"
          className="w-full max-w-[680px]"
          placeholder={isForgejoTokenSet ? "<hidden>" : ""}
          startContent={
            isForgejoTokenSet && (
              <KeyStatusIcon
                testId="forgejo-set-token-indicator"
                isSet={isForgejoTokenSet}
              />
            )
          }
        />
        {isForgejoTokenSet && onClear && (
          <BrandButton
            testId="forgejo-clear-token-button"
            type="button"
            variant="secondary"
            onClick={onClear}
          >
            {t(I18nKey.GIT$CLEAR_TOKEN)}
          </BrandButton>
        )}
      </div>

      <SettingsInput
        onChange={onForgejoHostChange || (() => {})}
        name="forgejo-host-input"
        testId="forgejo-host-input"
        label={t(I18nKey.FORGEJO$HOST_LABEL)}
        type="text"
        className="w-full max-w-[680px]"
        placeholder="codeberg.org"
        defaultValue={forgejoHostSet || undefined}
        startContent={
          forgejoHostSet &&
          forgejoHostSet.trim() !== "" && (
            <KeyStatusIcon testId="forgejo-set-host-indicator" isSet />
          )
        }
      />
    </div>
  );
}
