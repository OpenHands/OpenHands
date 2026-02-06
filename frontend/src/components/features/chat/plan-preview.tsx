import { useMemo, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { ArrowUpRight } from "lucide-react";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { USE_PLANNING_AGENT } from "#/utils/feature-flags";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";
import { cn } from "#/utils/utils";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import {
  planComponents,
  createPlanComponents,
} from "#/components/features/markdown/plan-components";
import { useScrollContext } from "#/context/scroll-context";

const MAX_CONTENT_LENGTH = 300;

// Shine effect class for streaming text
const SHINE_TEXT_CLASS = "shine-text";

// Plan components with shine effect applied for streaming state
const shineComponents = createPlanComponents(SHINE_TEXT_CLASS);

interface PlanPreviewProps {
  /** Raw plan content from PLAN.md file */
  planContent?: string | null;
  /** Whether the plan content is actively being streamed */
  isStreaming?: boolean;
  /** Whether the Build button should be disabled (e.g., while streaming) */
  isBuildDisabled?: boolean;
}

/* eslint-disable i18next/no-literal-string */
export function PlanPreview({
  planContent,
  isStreaming,
  isBuildDisabled,
}: PlanPreviewProps) {
  const { t } = useTranslation();
  const { navigateToTab } = useSelectConversationTab();
  const { handleBuildPlanClick } = useHandleBuildPlanClick();
  const { scrollDomToBottom } = useScrollContext();

  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  const handleViewClick = () => {
    navigateToTab("planner");
  };

  // Handle Build action with scroll to bottom
  const handleBuildClick = useCallback(
    (event?: React.MouseEvent<HTMLButtonElement>) => {
      handleBuildPlanClick(event);
      scrollDomToBottom();
    },
    [handleBuildPlanClick, scrollDomToBottom],
  );

  // Truncate plan content for preview
  const truncatedContent = useMemo(() => {
    if (!planContent) return "";
    if (planContent.length <= MAX_CONTENT_LENGTH) return planContent;
    return `${planContent.slice(0, MAX_CONTENT_LENGTH)}...`;
  }, [planContent]);

  if (!shouldUsePlanningAgent) {
    return null;
  }

  return (
    <div className="bg-[#25272d] border border-[#597FF4] rounded-[12px] w-full mt-2">
      {/* Header */}
      <div className="border-b border-[#525252] flex h-[41px] items-center px-2 gap-1">
        <LessonPlanIcon width={18} height={18} color="#9299aa" />
        <Typography.Text className="font-medium text-[11px] text-white tracking-[0.11px] leading-4">
          {t(I18nKey.COMMON$PLAN_MD)}
        </Typography.Text>
        <div className="flex-1" />
        <button
          type="button"
          onClick={handleViewClick}
          className="flex items-center gap-1 hover:opacity-80 transition-opacity cursor-pointer"
          data-testid="plan-preview-view-button"
        >
          <Typography.Text className="font-medium text-[11px] text-white tracking-[0.11px] leading-4">
            {t(I18nKey.COMMON$VIEW)}
          </Typography.Text>
          <ArrowUpRight className="text-white" size={18} />
        </button>
      </div>

      {/* Content */}
      <div
        data-testid="plan-preview-content"
        className="flex flex-col gap-[10px] p-4 text-[15px] text-white leading-[29px]"
      >
        {truncatedContent && (
          <>
            <MarkdownRenderer
              includeStandard
              components={isStreaming ? shineComponents : planComponents}
            >
              {truncatedContent}
            </MarkdownRenderer>
            {planContent && planContent.length > MAX_CONTENT_LENGTH && (
              <button
                type="button"
                onClick={handleViewClick}
                className="text-[#4a67bd] cursor-pointer hover:underline text-left"
                data-testid="plan-preview-read-more-button"
              >
                {t(I18nKey.COMMON$READ_MORE)}
              </button>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-[#525252] flex h-[54px] items-center justify-start px-4">
        <button
          type="button"
          onClick={handleBuildClick}
          disabled={isBuildDisabled}
          className={cn(
            "bg-white flex items-center justify-center h-[26px] px-2 rounded-[4px] w-[93px] transition-opacity",
            isBuildDisabled
              ? "opacity-50 cursor-not-allowed"
              : "hover:opacity-90 cursor-pointer",
          )}
          data-testid="plan-preview-build-button"
        >
          <Typography.Text className="font-medium text-[14px] text-black leading-5">
            {t(I18nKey.COMMON$BUILD)}{" "}
            <Typography.Text className="font-medium text-black">
              ⌘↩
            </Typography.Text>
          </Typography.Text>
        </button>
      </div>
    </div>
  );
}
