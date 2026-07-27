import { useTranslation } from "react-i18next";
import BranchIcon from "#/icons/u-code-branch.svg?react";
import { constructBranchUrl, cn } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { GitExternalLinkIcon } from "./git-external-link-icon";
import { useSettings } from "#/hooks/query/use-settings";

interface GitControlBarBranchButtonProps {
  selectedBranch: string | null | undefined;
  selectedRepository: string | null | undefined;
  gitProvider: Provider | null | undefined;
}

export function GitControlBarBranchButton({
  selectedBranch,
  selectedRepository,
  gitProvider,
}: GitControlBarBranchButtonProps) {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();

  const providerHost = gitProvider
    ? settings?.provider_tokens_set[gitProvider]
    : null;

  // Render the linked styling only when we have enough info to build a URL.
  // Local-workspace fallbacks may give us a branch name without a known
  // provider — show the branch text in that case but skip the external link.
  const hasLinkableBranch =
    !!selectedBranch && !!selectedRepository && !!gitProvider;
  const branchUrl = hasLinkableBranch
    ? constructBranchUrl(
        gitProvider,
        selectedRepository,
        selectedBranch,
        providerHost,
      )
    : undefined;

  const buttonText = selectedBranch || t(I18nKey.COMMON$NO_BRANCH);

  return (
    <a
      href={hasLinkableBranch ? branchUrl : undefined}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2.5 py-1 rounded-[100px] w-fit flex-shrink-0 max-w-[200px] truncate relative",
        hasLinkableBranch
          ? "border border-[var(--oh-border)] bg-transparent hover:border-[var(--oh-border-subtle)] cursor-pointer"
          : "border border-[rgba(71,74,84,0.50)] bg-transparent cursor-not-allowed min-w-[108px]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center flex-shrink-0">
        <BranchIcon width={12} height={12} color="white" />
      </div>
      <div
        className="font-normal text-white text-sm leading-5 truncate"
        title={buttonText}
      >
        {buttonText}
      </div>
      {hasLinkableBranch && <GitExternalLinkIcon />}
    </a>
  );
}
