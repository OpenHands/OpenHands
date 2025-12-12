import { useTranslation } from "react-i18next";
import BranchIcon from "#/icons/u-code-branch.svg?react";
import { constructBranchUrl, cn } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { GitExternalLinkIcon } from "./git-external-link-icon";

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
  const { t } = useTranslation();

  const hasBranch = selectedBranch && selectedRepository && gitProvider;
  const branchUrl = hasBranch
    ? constructBranchUrl(gitProvider, selectedRepository, selectedBranch)
    : undefined;

  const buttonText = hasBranch ? selectedBranch : t(I18nKey.COMMON$NO_BRANCH);

  return (
    <a
      href={hasBranch ? branchUrl : undefined}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2.5 py-1 rounded-[100px] w-fit flex-shrink-0 max-w-[200px] truncate relative",
        hasBranch
          ? "border border-[#525252] bg-transparent hover:border-[#454545] cursor-pointer"
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
      {hasBranch && <GitExternalLinkIcon />}
    </a>
  );
}
