import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { GitHubTokenHelpAnchor } from "./github-token-help-anchor";
import { KeyStatusIcon } from "../key-status-icon";
import { BrandButton } from "../brand-button";
import { cn } from "#/utils/utils";

interface GitHubTokenInputProps {
  onChange: (value: string) => void;
  onGitHubHostChange: (value: string) => void;
  isGitHubTokenSet: boolean;
  name: string;
  githubHostSet: string | null | undefined;
  className?: string;
  onClear?: () => void;
}

export function GitHubTokenInput({
  onChange,
  onGitHubHostChange,
  isGitHubTokenSet,
  name,
  githubHostSet,
  className,
  onClear,
}: GitHubTokenInputProps) {
  const { t } = useTranslation();

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex items-end gap-2">
        <SettingsInput
          testId={name}
          name={name}
          onChange={onChange}
          label={t(I18nKey.GITHUB$TOKEN_LABEL)}
          type="password"
          className="w-full max-w-[680px]"
          placeholder={isGitHubTokenSet ? "<hidden>" : ""}
          startContent={
            isGitHubTokenSet && (
              <KeyStatusIcon
                testId="gh-set-token-indicator"
                isSet={isGitHubTokenSet}
              />
            )
          }
        />
        {isGitHubTokenSet && onClear && (
          <BrandButton
            testId="gh-clear-token-button"
            type="button"
            variant="secondary"
            onClick={onClear}
          >
            {t(I18nKey.GIT$CLEAR_TOKEN)}
          </BrandButton>
        )}
      </div>

      <SettingsInput
        onChange={onGitHubHostChange || (() => {})}
        name="github-host-input"
        testId="github-host-input"
        label={t(I18nKey.GITHUB$HOST_LABEL)}
        type="text"
        className="w-full max-w-[680px]"
        placeholder="github.com"
        defaultValue={githubHostSet || undefined}
        startContent={
          githubHostSet &&
          githubHostSet.trim() !== "" && (
            <KeyStatusIcon testId="gh-set-host-indicator" isSet />
          )
        }
      />

      <GitHubTokenHelpAnchor />
    </div>
  );
}
