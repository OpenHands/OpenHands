import { useTranslation } from "react-i18next";
import { GitControlBarRepoButton } from "./git-control-bar-repo-button";
import { GitControlBarBranchButton } from "./git-control-bar-branch-button";
import { GitControlBarPullButton } from "./git-control-bar-pull-button";
import { GitControlBarPushButton } from "./git-control-bar-push-button";
import { GitControlBarPrButton } from "./git-control-bar-pr-button";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { GitControlBarTooltipWrapper } from "./git-control-bar-tooltip-wrapper";

interface GitControlBarProps {
  onSuggestionsClick: (value: string) => void;
}

export function GitControlBar({ onSuggestionsClick }: GitControlBarProps) {
  const { t } = useTranslation();

  const { data: conversation } = useActiveConversation();
  const { repositoryInfo } = useTaskPolling();
  const webSocketStatus = useUnifiedWebSocketStatus();

  // Priority: conversation data > task data
  // This ensures we show repository info immediately from task, then transition to conversation data
  const selectedRepository =
    conversation?.selected_repository || repositoryInfo?.selectedRepository;
  const gitProvider = (conversation?.git_provider ||
    repositoryInfo?.gitProvider) as Provider;
  const selectedBranch =
    conversation?.selected_branch || repositoryInfo?.selectedBranch;

  const hasRepository = !!selectedRepository;

  // Enable buttons only when conversation exists and WS is connected
  const isConversationReady = !!conversation && webSocketStatus === "CONNECTED";

  return (
    <div className="flex flex-row items-center">
      <div className="flex flex-row gap-2.5 items-center overflow-x-auto flex-wrap md:flex-nowrap relative scrollbar-hide">
        <GitControlBarTooltipWrapper
          tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
          testId="git-control-bar-repo-button-tooltip"
          shouldShowTooltip={!hasRepository}
        >
          <GitControlBarRepoButton
            selectedRepository={selectedRepository}
            gitProvider={gitProvider}
          />
        </GitControlBarTooltipWrapper>

        <GitControlBarTooltipWrapper
          tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
          testId="git-control-bar-branch-button-tooltip"
          shouldShowTooltip={!hasRepository}
        >
          <GitControlBarBranchButton
            selectedBranch={selectedBranch}
            selectedRepository={selectedRepository}
            gitProvider={gitProvider}
          />
        </GitControlBarTooltipWrapper>

        {hasRepository ? (
          <>
            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-pull-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPullButton
                onSuggestionsClick={onSuggestionsClick}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>

            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-push-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPushButton
                onSuggestionsClick={onSuggestionsClick}
                hasRepository={hasRepository}
                currentGitProvider={gitProvider}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>

            <GitControlBarTooltipWrapper
              tooltipMessage={t(I18nKey.COMMON$GIT_TOOLS_DISABLED_CONTENT)}
              testId="git-control-bar-pr-button-tooltip"
              shouldShowTooltip={!hasRepository}
            >
              <GitControlBarPrButton
                onSuggestionsClick={onSuggestionsClick}
                hasRepository={hasRepository}
                currentGitProvider={gitProvider}
                isConversationReady={isConversationReady}
              />
            </GitControlBarTooltipWrapper>
          </>
        ) : null}
      </div>
    </div>
  );
}
