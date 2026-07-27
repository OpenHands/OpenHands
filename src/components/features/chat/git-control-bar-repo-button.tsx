import { FolderOpen } from "lucide-react";
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
  workspaceName?: string | null;
  emptyStateLabel?: string;
  onClick?: () => void;
  disabled?: boolean;
}

export function GitControlBarRepoButton({
  selectedRepository,
  gitProvider,
  workspaceName,
  emptyStateLabel: _emptyStateLabel,
  onClick,
  disabled,
}: GitControlBarRepoButtonProps) {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();

  // Render as an external link only when we know both the repo and the
  // provider (so the URL is well-defined). Local-workspace fallbacks may
  // populate `selectedRepository` without a provider; in that case we still
  // show the repo name but keep the button non-link styling.
  const hasLinkableRepo = !!selectedRepository && !!gitProvider;

  // Get the host for the current provider from settings
  const providerHost = gitProvider
    ? settings?.provider_tokens_set[gitProvider]
    : null;

  const repositoryUrl = hasLinkableRepo
    ? constructRepositoryUrl(gitProvider, selectedRepository, providerHost)
    : undefined;

  const showConnectRepoCta = !selectedRepository && !workspaceName;
  const buttonText =
    selectedRepository || workspaceName || t(I18nKey.COMMON$CONNECT_REPO);

  if (hasLinkableRepo) {
    return (
      <a
        href={repositoryUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2.5 py-1 rounded-[100px] flex-1 truncate relative",
          "border border-[var(--oh-border)] bg-transparent hover:border-[var(--oh-border-subtle)] cursor-pointer",
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
        "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2.5 py-1 rounded-[100px] truncate relative",
        "border border-[rgba(71,74,84,0.50)] bg-transparent",
        disabled
          ? "cursor-not-allowed opacity-50"
          : "cursor-pointer hover:border-[var(--oh-border-subtle)]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center flex-shrink-0 text-white">
        {showConnectRepoCta ? (
          <FolderOpen
            className="w-3 h-3"
            strokeWidth={2}
            aria-hidden
            data-testid="git-control-bar-connect-repo-icon"
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
    </button>
  );
}
