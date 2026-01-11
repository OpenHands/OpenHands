import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { useConversationStore } from "#/stores/conversation-store";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";

function PlannerTab() {
  const { t } = useTranslation();
  const { scrollRef: scrollContainerRef, onChatBodyScroll } = useScrollToBottom(
    React.useRef<HTMLDivElement>(null),
  );

  const { planContent } = useConversationStore();
  const { handlePlanClick } = useHandlePlanClick();

  if (planContent !== null && planContent !== undefined) {
    return (
      <div
        ref={scrollContainerRef}
        onScroll={(e) => onChatBodyScroll(e.currentTarget)}
        className="flex flex-col w-full h-full p-4 overflow-auto"
      >
        <MarkdownRenderer includeStandard includeHeadings>
          {planContent}
        </MarkdownRenderer>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-10">
      <LessonPlanIcon width={109} height={109} className="text-muted" />
      <span className="text-subtle text-[19px] font-normal leading-5 pb-9">
        {t(I18nKey.PLANNER$EMPTY_MESSAGE)}
      </span>
      <button
        type="button"
        onClick={handlePlanClick}
        className="flex w-[164px] h-[40px] p-2 justify-center items-center shrink-0 rounded-lg bg-content overflow-hidden text-base text-ellipsis font-sans text-[16px] not-italic font-normal leading-[20px] hover:cursor-pointer hover:opacity-80"
      >
        {t(I18nKey.COMMON$CREATE_A_PLAN)}
      </button>
    </div>
  );
}

export default PlannerTab;
