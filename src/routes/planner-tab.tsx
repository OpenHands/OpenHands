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
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";

function PlannerTab() {
  const { t } = useTranslation("openhands");
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const {
    scrollRef: scrollContainerRef,
    onChatBodyScroll,
    autoScroll,
    scrollDomToBottom,
  } = useScrollToBottom(scrollRef);

  const { planContent, conversationMode } = useConversationStore();

  // Auto-scroll to bottom when plan content changes
  React.useEffect(() => {
    if (autoScroll) {
      scrollDomToBottom();
    }
  }, [planContent, autoScroll, scrollDomToBottom]);

  const [isCopy, setIsCopy] = React.useState(false);

  const handleCopyToClipboard = async () => {
    if (planContent) {
      await navigator.clipboard.writeText(planContent);
      setIsCopy(true);
    }
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
  const isPlanMode = conversationMode === "plan";
  const { handlePlanClick } = useHandlePlanClick();

  if (planContent !== null && planContent !== undefined) {
    return (
      <div className="relative w-full h-full">
        <div className="absolute top-2 right-4 z-10 rounded-md bg-tertiary">
          <CopyToClipboardButton
            isHidden={false}
            isDisabled={isCopy}
            onClick={handleCopyToClipboard}
            mode={isCopy ? "copied" : "copy"}
          />
        </div>
        <div
          ref={scrollContainerRef}
          onScroll={(e) => onChatBodyScroll(e.currentTarget)}
          className="flex flex-col w-full h-full p-4 overflow-auto"
        >
          <MarkdownRenderer includeStandard components={planComponents}>
            {planContent}
          </MarkdownRenderer>
        </div>
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
          isDisabled={isPlanMode}
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
