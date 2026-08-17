import { FaBitbucket, FaGithub, FaGitlab } from "react-icons/fa6";
import { FaCodeBranch } from "react-icons/fa";
import { IconType } from "react-icons/lib";
import { RepositorySelection } from "#/api/open-hands.types";
import { Provider } from "#/types/settings";
import AzureDevOpsLogo from "#/assets/branding/azure-devops-logo.svg?react";
import { cn } from "#/utils/utils";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "./conversation-card-meta-chip";

/**
 * Repo and branch share one ``overflow-hidden`` row, so unlike the standalone
 * chips they must be allowed to shrink: with the shared class's ``shrink-0``
 * a long repository name claims the full row and pushes the branch chip out of
 * view entirely. Shrinking lets both stay visible and truncate proportionally.
 */
const REPO_LINK_CHIP_CLASSNAME = cn(
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  "shrink",
);

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
  const repository = selectedRepository.selected_repository;
  const branch = selectedRepository.selected_branch;

  return (
    <div className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
      {repository ? (
        <span
          data-testid="conversation-card-selected-repository"
          title={repository}
          className={REPO_LINK_CHIP_CLASSNAME}
        >
          {(Icon || selectedRepository.git_provider === "azure_devops") && (
            <span
              className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
              aria-hidden
            >
              {Icon ? (
                <Icon className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
              ) : (
                <AzureDevOpsLogo
                  className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
                />
              )}
            </span>
          )}
          <span className="truncate leading-4">{repository}</span>
        </span>
      ) : null}
      {branch ? (
        <span
          data-testid="conversation-card-selected-branch"
          title={branch}
          className={REPO_LINK_CHIP_CLASSNAME}
        >
          <span
            className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
            aria-hidden
          >
            <FaCodeBranch
              className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
            />
          </span>
          <span className="truncate leading-4">{branch}</span>
        </span>
      ) : null}
    </div>
  );
}
