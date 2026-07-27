import { FaBitbucket, FaGithub, FaGitlab } from "react-icons/fa6";
import { FaCodeBranch } from "react-icons/fa";
import { IconType } from "react-icons/lib";
import { RepositorySelection } from "#/api/open-hands.types";
import { Provider } from "#/types/settings";
import AzureDevOpsLogo from "#/assets/branding/azure-devops-logo.svg?react";

interface ConversationRepoLinkProps {
  selectedRepository: RepositorySelection;
}

const providerIcon: Partial<Record<Provider, IconType>> = {
  bitbucket: FaBitbucket,
  bitbucket_data_center: FaBitbucket,
  github: FaGithub,
  gitlab: FaGitlab,
};

export function ConversationRepoLink({
  selectedRepository,
}: ConversationRepoLinkProps) {
  const Icon = selectedRepository.git_provider
    ? providerIcon[selectedRepository.git_provider]
    : null;

  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
      <div className="flex min-w-0 items-center gap-1">
        {Icon && <Icon size={14} className="shrink-0 text-[var(--oh-muted)]" />}
        {selectedRepository.git_provider === "azure_devops" && (
          <AzureDevOpsLogo className="h-[14px] w-[14px] shrink-0 text-[var(--oh-muted)]" />
        )}
        <span
          data-testid="conversation-card-selected-repository"
          className="min-w-0 truncate text-xs text-[var(--oh-muted)]"
        >
          {selectedRepository.selected_repository}
        </span>
      </div>
      <div className="flex min-w-0 items-center gap-1">
        <FaCodeBranch size={12} className="shrink-0 text-[var(--oh-muted)]" />

        <span
          data-testid="conversation-card-selected-branch"
          className="min-w-0 truncate text-xs text-[var(--oh-muted)]"
        >
          {selectedRepository.selected_branch}
        </span>
      </div>
    </div>
  );
}
