import { useTranslation } from "react-i18next";
import { CommitList } from "#/components/features/diff-viewer/commit-list";
import { DiffDrawerIcon } from "#/components/features/diff-viewer/diff-drawer-icon";
import { useUnifiedGitCommits } from "#/hooks/query/use-unified-git-commits";
import { useConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { RuntimeWaitingState } from "#/components/features/conversation-panel/runtime-waiting-state";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";

/**
 * The Files tab's "Commits" view: the workspace's recent commit history
 * (newest first), each commit expandable into its per-file diffs. Sits
 * behind the third segment of the Diff/Files toggle, which is only
 * offered when the agent server supports the commits API.
 */
function GitCommits() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const { commits, hasMore, isUnsupported, isLoading, isSuccess } =
    useUnifiedGitCommits();

  const { curAgentState } = useAgentState();
  const runtimeIsActive = !RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const showList = isSuccess && !isUnsupported && commits.length > 0;

  return (
    <main className="h-full w-full flex flex-col items-stretch">
      {showList ? (
        <div className="h-full overflow-y-auto flex flex-col items-stretch custom-scrollbar-always">
          {/* Keyed by conversation so switching conversations collapses any
              expanded commit. */}
          <CommitList
            key={conversationId}
            commits={commits}
            hasMore={hasMore}
          />
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          {!runtimeIsActive && (
            <RuntimeWaitingState
              testId="commits-tab-status"
              messageKey={I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME}
            />
          )}
          {runtimeIsActive && isLoading && (
            <RuntimeWaitingState
              testId="commits-tab-status"
              messageKey={I18nKey.DIFF_VIEWER$LOADING}
            />
          )}
          {runtimeIsActive && !isLoading && (
            <ConversationTabEmptyState icon={<DiffDrawerIcon />}>
              {t(I18nKey.DIFF_VIEWER$NO_COMMITS)}
            </ConversationTabEmptyState>
          )}
        </div>
      )}
    </main>
  );
}

export default GitCommits;
