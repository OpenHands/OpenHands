import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { CommitList } from "#/components/features/diff-viewer/commit-list";
import { DiffDrawerIcon } from "#/components/features/diff-viewer/diff-drawer-icon";
import { useUnifiedGitCommits } from "#/hooks/query/use-unified-git-commits";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { useConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import { RuntimeWaitingState } from "#/components/features/conversation-panel/runtime-waiting-state";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useConversationStore } from "#/stores/conversation-store";

/**
 * The Files tab's "Commits" view: the workspace's recent commit history
 * (newest first), each commit expandable into its per-file diffs. Sits
 * behind the third segment of the Diff/Files toggle, which is only
 * offered when the agent server supports the commits API.
 */
function GitCommits() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const {
    commits,
    hasMore,
    isWaitingForResume,
    isWaitingForRuntime,
    isUnavailable,
    isLoading,
    refetch,
  } = useUnifiedGitCommits();
  const {
    data: uncommittedChanges,
    isSuccess: uncommittedSuccess,
    isLoading: uncommittedLoading,
  } = useUnifiedGetGitChanges();
  const commitsAutoExpandSection = useConversationStore(
    (state) => state.commitsAutoExpandSection,
  );
  const setCommitsAutoExpandSection = useConversationStore(
    (state) => state.setCommitsAutoExpandSection,
  );
  const handleAutoExpandHandled = useCallback(
    () => setCommitsAutoExpandSection(null),
    [setCommitsAutoExpandSection],
  );

  const hasUncommitted =
    uncommittedSuccess && (uncommittedChanges?.length ?? 0) > 0;
  const showList = commits.length > 0 || hasUncommitted;
  const showStatus = isUnavailable || isWaitingForResume || !showList;
  const isWaiting =
    !isUnavailable &&
    !isWaitingForResume &&
    (isWaitingForRuntime || isLoading || uncommittedLoading);

  return (
    <main className="h-full w-full overflow-y-auto custom-scrollbar-always">
      {showList && (
        <CommitList
          key={conversationId}
          commits={commits}
          hasMore={hasMore}
          uncommittedChanges={
            uncommittedSuccess ? (uncommittedChanges ?? []).slice(0, 100) : []
          }
          autoExpandUncommitted={commitsAutoExpandSection === "uncommitted"}
          onAutoExpandHandled={handleAutoExpandHandled}
        />
      )}
      {showStatus &&
        (isWaiting ? (
          <RuntimeWaitingState
            testId="commits-tab-status"
            messageKey={
              isWaitingForRuntime
                ? I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME
                : I18nKey.DIFF_VIEWER$LOADING
            }
          />
        ) : (
          <ConversationTabEmptyState
            icon={<DiffDrawerIcon />}
            className={showList ? "h-auto" : undefined}
            action={
              isUnavailable && (
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => void refetch()}
                >
                  {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
                </BrandButton>
              )
            }
          >
            {isUnavailable ? (
              <span role="alert">
                {t(I18nKey.DIFF_VIEWER$COMMITS_UNAVAILABLE)}
              </span>
            ) : (
              t(
                isWaitingForResume
                  ? I18nKey.COMMON$PAUSED
                  : I18nKey.DIFF_VIEWER$NO_COMMITS,
              )
            )}
          </ConversationTabEmptyState>
        ))}
    </main>
  );
}

export default GitCommits;
