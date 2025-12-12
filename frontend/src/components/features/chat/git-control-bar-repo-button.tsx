import { useTranslation } from "react-i18next";
import { constructRepositoryUrl, cn } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { GitExternalLinkIcon } from "./git-external-link-icon";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";

interface GitControlBarRepoButtonProps {
  selectedRepository: string | null | undefined;
  gitProvider: Provider | null | undefined;
}

export function GitControlBarRepoButton({
  selectedRepository,
  gitProvider,
}: GitControlBarRepoButtonProps) {
  const { t } = useTranslation();

  const hasRepository = selectedRepository && gitProvider;

  const repositoryUrl = hasRepository
    ? constructRepositoryUrl(gitProvider, selectedRepository)
    : undefined;

  const buttonText = hasRepository
    ? selectedRepository
    : t(I18nKey.COMMON$NO_REPO_CONNECTED);

  return (
    <a
      href={hasRepository ? repositoryUrl : undefined}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2.5 py-1 rounded-[100px] flex-1 truncate relative",
        hasRepository
          ? "border border-[#525252] bg-transparent hover:border-[#454545] cursor-pointer"
          : "border border-[rgba(71,74,84,0.50)] bg-transparent cursor-not-allowed min-w-[170px]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center flex-shrink-0">
        {hasRepository ? (
          <GitProviderIcon
            gitProvider={gitProvider as Provider}
            className="w-3 h-3 inline-flex"
          />
        ) : (
          <RepoForkedIcon width={12} height={12} color="white" />
        )}
      </div>
      <div
        className="font-normal text-white text-sm leading-5 truncate flex-1 min-w-0"
        title={buttonText}
      >
        {buttonText}
      </div>
      {hasRepository && <GitExternalLinkIcon />}
    </a>
  );
}
