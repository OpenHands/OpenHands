import React from "react";
import { ExternalLink, GitBranch, GitCommitHorizontal } from "lucide-react";
import { FaCodeBranch } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import PrIcon from "#/icons/u-pr.svg?react";
import { GitProviderIcon } from "#/components/shared/git-provider-icon";
import { useRepositoryPullRequests } from "#/hooks/query/use-repository-git-items";
import { useUnifiedGitCommits } from "#/hooks/query/use-unified-git-commits";
import { useConversationPrimaryRepository } from "#/hooks/use-conversation-primary-repository";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useSettings } from "#/hooks/query/use-settings";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { I18nKey } from "#/i18n/declaration";
import { cn, constructBranchUrl, constructRepositoryUrl } from "#/utils/utils";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";
import { CONVERSATION_OVERVIEW_DRAWER_SECTION } from "./conversation-overview-drawer.types";
import {
  CONVERSATION_OVERVIEW_GIT_PART,
  isOverviewGitPartPinned,
} from "./conversation-overview-sections";

const ROW_ICON_CLASSNAME = "size-4 shrink-0 text-[var(--oh-muted)]";
const EXTERNAL_LINK_ICON_CLASSNAME = cn(
  "size-3.5 shrink-0 text-[var(--oh-muted)]",
  "opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100",
);
// Non-localizable empty-count glyph (zero PRs).

const ZERO_COUNT_LABEL = "-";

/**
 * Repo / branch / commits / PR rows for the overview git block.
 * Changes is rendered by the parent panel ahead of this section.
 */
export function ConversationOverviewGitSection() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const { state } = useConversationLocalStorageState(conversationId);
  const { navigateToCommits } = useSelectConversationTab();
  const overviewDrawer = useConversationOverviewDrawerOptional();
  const { data: settings } = useSettings();
  const { repository, provider, branch, isConnected } =
    useConversationPrimaryRepository();

  const showRepository = isOverviewGitPartPinned(
    CONVERSATION_OVERVIEW_GIT_PART.repository,
    state.unpinnedOverviewGitParts ?? [],
  );
  const showBranch = isOverviewGitPartPinned(
    CONVERSATION_OVERVIEW_GIT_PART.branch,
    state.unpinnedOverviewGitParts ?? [],
  );
  const showCommits = isOverviewGitPartPinned(
    CONVERSATION_OVERVIEW_GIT_PART.commits,
    state.unpinnedOverviewGitParts ?? [],
  );
  const showPullRequests = isOverviewGitPartPinned(
    CONVERSATION_OVERVIEW_GIT_PART.pull_requests,
    state.unpinnedOverviewGitParts ?? [],
  );

  const pullRequestsQuery = useRepositoryPullRequests(
    showPullRequests ? repository : null,
    showPullRequests ? provider : null,
  );
  // Same page the Files → Commits drawer uses; no separate total from the
  // API, so we surface the fetched count (and a trailing + when capped).
  const commitsQuery = useUnifiedGitCommits();

  if (!isConnected || !repository || !provider) {
    return null;
  }

  if (!showRepository && !showBranch && !showCommits && !showPullRequests) {
    return null;
  }

  const providerHost = settings?.provider_tokens_set?.[provider] ?? null;
  const repoUrl = constructRepositoryUrl(provider, repository, providerHost);
  const branchUrl = branch
    ? constructBranchUrl(provider, repository, branch, providerHost)
    : null;
  const pullRequestsCount = pullRequestsQuery.data?.length ?? null;
  const commitsCount = commitsQuery.isUnsupported
    ? null
    : commitsQuery.commits.length;
  const commitsCountLabel =
    commitsCount !== null && commitsCount > 0
      ? commitsQuery.hasMore
        ? `${commitsCount}+`
        : String(commitsCount)
      : ZERO_COUNT_LABEL;

  const openPullRequests = () => {
    overviewDrawer?.openSection(
      CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests,
    );
  };

  const openCommits = () => {
    overviewDrawer?.closeDrawer();
    navigateToCommits();
  };

  const showHeader = showRepository || showBranch;
  const showList = showCommits || showPullRequests;

  return (
    <div data-testid="conversation-overview-git-section">
      {showHeader ? (
        <div className="px-2 pb-1">
          {showRepository ? (
            <a
              href={repoUrl}
              target="_blank"
              rel="noreferrer"
              data-testid="conversation-overview-git-repo"
              className={cn(
                "group flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5",
                "transition-colors hover:bg-white/5",
              )}
            >
              <GitProviderIcon
                gitProvider={provider}
                className={ROW_ICON_CLASSNAME}
              />
              <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
                {repository}
              </span>
              <ExternalLink
                className={EXTERNAL_LINK_ICON_CLASSNAME}
                aria-hidden
              />
            </a>
          ) : null}
          {showBranch ? (
            branch && branchUrl ? (
              <a
                href={branchUrl}
                target="_blank"
                rel="noreferrer"
                data-testid="conversation-overview-git-branch"
                className={cn(
                  "group flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5",
                  "transition-colors hover:bg-white/5",
                )}
              >
                <FaCodeBranch
                  size={14}
                  className="shrink-0 text-[var(--oh-muted)]"
                  aria-hidden
                />
                <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
                  {branch}
                </span>
                <ExternalLink
                  className={EXTERNAL_LINK_ICON_CLASSNAME}
                  aria-hidden
                />
              </a>
            ) : (
              <div
                data-testid="conversation-overview-git-branch"
                className="flex min-w-0 items-center gap-2 px-2 py-1.5"
              >
                <GitBranch
                  className="size-3.5 shrink-0 text-[var(--oh-muted)]"
                  aria-hidden
                />
                <span
                  className={cn(
                    "text-sm",
                    branch
                      ? "text-[var(--oh-foreground)]"
                      : "text-[var(--oh-muted)]",
                  )}
                >
                  {branch || t(I18nKey.CONVERSATION$OVERVIEW_NONE)}
                </span>
              </div>
            )
          ) : null}
        </div>
      ) : null}

      {showList ? (
        <ul className="px-2">
          {showCommits ? (
            <li>
              <button
                type="button"
                data-testid="conversation-overview-commits"
                onClick={openCommits}
                className={cn(
                  "flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5",
                  "cursor-pointer bg-transparent text-left transition-colors hover:bg-white/5",
                )}
              >
                <GitCommitHorizontal
                  className={ROW_ICON_CLASSNAME}
                  aria-hidden
                />
                <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
                  {t(I18nKey.DIFF_VIEWER$COMMITS)}
                </span>
                {commitsQuery.isLoading ? (
                  <span className="text-sm tabular-nums text-[var(--oh-muted)]">
                    …
                  </span>
                ) : (
                  <span
                    data-testid="conversation-overview-commits-count"
                    className="text-sm tabular-nums text-[var(--oh-muted)]"
                  >
                    {commitsCountLabel}
                  </span>
                )}
              </button>
            </li>
          ) : null}
          {showPullRequests ? (
            <li>
              <button
                type="button"
                data-testid="conversation-overview-pull-requests"
                onClick={openPullRequests}
                className={cn(
                  "flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5",
                  "cursor-pointer bg-transparent text-left transition-colors hover:bg-white/5",
                )}
              >
                <PrIcon className={ROW_ICON_CLASSNAME} aria-hidden />
                <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
                  {t(I18nKey.CONVERSATION$OVERVIEW_PULL_REQUESTS)}
                </span>
                {pullRequestsQuery.isLoading ? (
                  <span className="text-sm tabular-nums text-[var(--oh-muted)]">
                    …
                  </span>
                ) : (
                  <span
                    data-testid="conversation-overview-pull-requests-count"
                    className="text-sm tabular-nums text-[var(--oh-muted)]"
                  >
                    {pullRequestsCount !== null && pullRequestsCount > 0
                      ? pullRequestsCount
                      : ZERO_COUNT_LABEL}
                  </span>
                )}
              </button>
            </li>
          ) : null}
        </ul>
      ) : null}
    </div>
  );
}
