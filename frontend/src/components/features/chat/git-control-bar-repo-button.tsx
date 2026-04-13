import { useTranslation } from "react-i18next";
import { constructRepositoryUrl, cn } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { GitExternalLinkIcon } from "./git-external-link-icon";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { useSettings } from "#/hooks/query/use-settings";

interface GitControlBarRepoButtonProps {
  selectedRepository: string | null | undefined;
  gitProvider: Provider | null | undefined;
  onClick?: () => void;
  disabled?: boolean;
}

export function GitControlBarRepoButton({
  selectedRepository,
  gitProvider,
  onClick,
  disabled,
}: GitControlBarRepoButtonProps) {
  const { t } = useTranslation();
  const { data: settings } = useSettings();

  const hasRepository = selectedRepository && gitProvider;

  // Get the host for the current provider from settings
  const providerHost = gitProvider
    ? settings?.provider_tokens_set[gitProvider]
    : null;

  const repositoryUrl = hasRepository
    ? constructRepositoryUrl(gitProvider, selectedRepository, providerHost)
    : undefined;

  const buttonText = hasRepository
    ? selectedRepository
    : t(I18nKey.COMMON$NO_REPO_CONNECTED);

  if (hasRepository) {
    return (
      <a
        href={repositoryUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "group inline-flex min-w-0 max-w-[min(70vw,20rem)] flex-row items-center justify-between gap-2 rounded-[100px] border border-[#525252] bg-transparent py-1 pl-2.5 pr-2.5 hover:border-[#454545]",
          "cursor-pointer",
        )}
      >
        <div className="w-3 h-3 flex items-center justify-center flex-shrink-0">
          <GitProviderIcon
            gitProvider={gitProvider as Provider}
            className="w-3 h-3 inline-flex"
          />
        </div>
        <div
          className="font-normal text-white text-sm leading-5 truncate flex-1 min-w-0"
          title={buttonText}
        >
          {buttonText}
        </div>
        <GitExternalLinkIcon />
      </a>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "group inline-flex min-w-[150px] max-w-[min(70vw,20rem)] flex-row items-center justify-between gap-2 rounded-[100px] border border-[rgba(71,74,84,0.50)] bg-transparent py-1 pl-2.5 pr-2.5",
        disabled
          ? "cursor-not-allowed opacity-50"
          : "cursor-pointer hover:border-[#454545]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center flex-shrink-0">
        <RepoForkedIcon width={12} height={12} color="white" />
      </div>
      <div
        className="font-normal text-white text-sm leading-5 truncate flex-1 min-w-0"
        title={buttonText}
      >
        {buttonText}
      </div>
    </button>
  );
}
