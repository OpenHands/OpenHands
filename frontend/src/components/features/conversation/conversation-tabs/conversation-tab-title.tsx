import React from "react";
import { useTranslation } from "react-i18next";
import RefreshIcon from "#/icons/u-refresh.svg?react";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";
import { useAgentState } from "#/hooks/use-agent-state";
import { useConversationStore } from "#/stores/conversation-store";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

type ConversationTabTitleProps = {
  title: string;
  conversationKey: string;
};

/* eslint-disable i18next/no-literal-string */
export function ConversationTabTitle({
  title,
  conversationKey,
}: ConversationTabTitleProps) {
  const { t } = useTranslation();
  const { refetch, isFetching } = useUnifiedGetGitChanges();
  const { handleBuildPlanClick } = useHandleBuildPlanClick();
  const { curAgentState } = useAgentState();
  const { planContent } = useConversationStore();
  const [isCopy, setIsCopy] = React.useState(false);

  const handleRefresh = () => {
    refetch();
  };

  // Determine if Build button should be disabled
  const isAgentRunning =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;
  const isBuildDisabled = isAgentRunning || !planContent;

  const handleCopyPlanClick = async () => {
    if (!planContent) {
      return;
    }

    await navigator.clipboard.writeText(planContent);
    setIsCopy(true);
  };

  React.useEffect(() => {
    let timeout: NodeJS.Timeout;

    if (isCopy) {
      timeout = setTimeout(() => {
        setIsCopy(false);
      }, 2000);
    }

    return () => {
      clearTimeout(timeout);
    };
  }, [isCopy]);

  return (
    <div className="flex flex-row items-center justify-between border-b border-[#474A54] py-2 px-3">
      <span className="text-xs font-medium text-white">{title}</span>
      {conversationKey === "editor" && (
        <button
          type="button"
          className="flex w-[26px] py-1 justify-center items-center gap-[10px] rounded-[7px] hover:enabled:bg-[#474A54] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleRefresh}
          disabled={isFetching}
        >
          <RefreshIcon
            width={12.75}
            height={15}
            color="#ffffff"
            className={isFetching ? "animate-spin" : ""}
          />
        </button>
      )}
      {conversationKey === "planner" && (
        <div className="flex items-center gap-1">
          {planContent && (
            <CopyToClipboardButton
              isHidden={false}
              isDisabled={isCopy}
              onClick={handleCopyPlanClick}
              mode={isCopy ? "copied" : "copy"}
            />
          )}
          <button
            type="button"
            onClick={handleBuildPlanClick}
            disabled={isBuildDisabled}
            className={cn(
              "flex items-center justify-center h-5 min-w-17 px-2 rounded bg-white transition-opacity",
              isBuildDisabled
                ? "opacity-50 cursor-not-allowed"
                : "hover:opacity-90 cursor-pointer",
            )}
            data-testid="planner-tab-build-button"
          >
            <Typography.Text className="text-black text-[11px] font-medium leading-5">
              {t(I18nKey.COMMON$BUILD)} ⌘↩
            </Typography.Text>
          </button>
        </div>
      )}
    </div>
  );
}
