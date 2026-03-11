import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { BitbucketDCTokenHelpAnchor } from "./bitbucket-dc-token-help-anchor";
import { KeyStatusIcon } from "../key-status-icon";
import { cn } from "#/utils/utils";

interface BitbucketDCTokenInputProps {
  onChange: (value: string) => void;
  onBitbucketDCHostChange: (value: string) => void;
  isBitbucketDCTokenSet: boolean;
  name: string;
  bitbucketDCHostSet: string | null | undefined;
  className?: string;
}

export function BitbucketDCTokenInput({
  onChange,
  onBitbucketDCHostChange,
  isBitbucketDCTokenSet,
  name,
  bitbucketDCHostSet,
  className,
}: BitbucketDCTokenInputProps) {
  const { t } = useTranslation();

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <SettingsInput
        testId={name}
        name={name}
        onChange={onChange}
        label={t(I18nKey.BITBUCKET_DATA_CENTER$TOKEN_LABEL)}
        type="password"
        className="w-full max-w-[680px]"
        placeholder={isBitbucketDCTokenSet ? "<hidden>" : "username:token"}
        startContent={
          isBitbucketDCTokenSet && (
            <KeyStatusIcon
              testId="bb-dc-set-token-indicator"
              isSet={isBitbucketDCTokenSet}
            />
          )
        }
      />

      <SettingsInput
        onChange={onBitbucketDCHostChange || (() => {})}
        name="bitbucket-dc-host-input"
        testId="bitbucket-dc-host-input"
        label={t(I18nKey.BITBUCKET_DATA_CENTER$HOST_LABEL)}
        type="text"
        className="w-full max-w-[680px]"
        placeholder="bitbucket.your-company.com"
        defaultValue={bitbucketDCHostSet || undefined}
        startContent={
          bitbucketDCHostSet &&
          bitbucketDCHostSet.trim() !== "" && (
            <KeyStatusIcon testId="bb-dc-set-host-indicator" isSet />
          )
        }
      />

      <BitbucketDCTokenHelpAnchor />
    </div>
  );
}
