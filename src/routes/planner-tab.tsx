import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ListTodo } from "lucide-react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";
import { useConversationStore } from "#/stores/conversation-store";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { planComponents } from "#/components/features/markdown/plan-components";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useReadConversationFile } from "#/hooks/mutation/use-read-conversation-file";

function PlannerTab() {
  const { t } = useTranslation("openhands");
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const {
    scrollRef: scrollContainerRef,
    onChatBodyScroll,
    autoScroll,
    scrollDomToBottom,
  } = useScrollToBottom(scrollRef);

  const {
    planContent,
    conversationMode,
    localPlanningConversationId,
    setPlanContent,
  } = useConversationStore();
  const { data: conversation } = useActiveConversation();
  const { mutate: readConversationFile } = useReadConversationFile();

  React.useEffect(() => {
    if (!conversation?.id || planContent) return;

    readConversationFile(
      { conversationId: conversation.id },
      {
        onSuccess: (fileContent) => {
          setPlanContent(fileContent);
        },
        onError: () => {
          // Missing PLAN.md is the normal empty-planner state.
        },
      },
    );
  }, [conversation?.id, planContent, readConversationFile, setPlanContent]);

  // Auto-scroll to bottom when plan content changes
  React.useEffect(() => {
    if (autoScroll) {
      scrollDomToBottom();
    }
  }, [planContent, autoScroll, scrollDomToBottom]);
  const isPlanMode = conversationMode === "plan";
  const { handlePlanClick, isCreatingConversation } = useHandlePlanClick();

  if (planContent !== null && planContent !== undefined) {
    return (
      <div
        ref={scrollContainerRef}
        onScroll={(e) => onChatBodyScroll(e.currentTarget)}
        className="flex flex-col w-full h-full p-4 overflow-auto"
      >
        <MarkdownRenderer includeStandard components={planComponents}>
          {planContent}
        </MarkdownRenderer>
      </div>
    );
  }

  return (
    <ConversationTabEmptyState
      icon={<ListTodo aria-hidden strokeWidth={2} className="size-full" />}
      action={
        <BrandButton
          type="button"
          variant="secondary"
          onClick={handlePlanClick}
          isDisabled={
            isPlanMode ||
            isCreatingConversation ||
            !!localPlanningConversationId
          }
          className="min-w-40 justify-center px-6"
        >
          {t(I18nKey.COMMON$CREATE_A_PLAN)}
        </BrandButton>
      }
    >
      {t(I18nKey.PLANNER$EMPTY_MESSAGE)}
    </ConversationTabEmptyState>
  );
}

export default PlannerTab;
