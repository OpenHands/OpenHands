import { useTranslation } from "react-i18next";
import { ArrowUpRight } from "lucide-react";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { USE_PLANNING_AGENT } from "#/utils/feature-flags";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

interface PlanPreviewProps {
  title?: string;
  description?: string;
  onViewClick?: () => void;
  onBuildClick?: () => void;
}

// TODO: Remove the hardcoded values and use the plan content from the conversation store
/* eslint-disable i18next/no-literal-string */
export function PlanPreview({
  title = "Improve Developer Onboarding and Examples",
  description = "Based on the analysis of Browser-Use's current documentation and examples, this plan addresses gaps in developer onboarding by creating a progressive learning path, troubleshooting resources, and practical examples that address real-world scenarios (like the LM Studio/local LLM integration issues encountered...",
  onViewClick,
  onBuildClick,
}: PlanPreviewProps) {
  const { t } = useTranslation();

  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  if (!shouldUsePlanningAgent) {
    return null;
  }

  return (
    <div className="bg-[#25272d] border border-[#597FF4] rounded-[12px] w-full mb-4 mt-2">
      {/* Header */}
      <div className="border-b border-[#525252] flex h-[41px] items-center px-2 gap-1">
        <LessonPlanIcon width={18} height={18} color="#9299aa" />
        <Typography.Text className="font-medium text-[11px] text-white tracking-[0.11px] leading-4">
          {t(I18nKey.COMMON$PLAN_MD)}
        </Typography.Text>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onViewClick}
          className="flex items-center gap-1 hover:opacity-80 transition-opacity"
        >
          <Typography.Text className="font-medium text-[11px] text-white tracking-[0.11px] leading-4">
            {t(I18nKey.COMMON$VIEW)}
          </Typography.Text>
          <ArrowUpRight className="text-white" size={18} />
        </button>
      </div>

      {/* Content */}
      <div className="flex flex-col gap-[10px] p-4">
        <h3 className="font-bold text-[19px] text-white leading-[29px]">
          {title}
        </h3>
        <p className="text-[15px] text-white leading-[29px]">
          {description}
          <Typography.Text className="text-[#4a67bd] cursor-pointer hover:underline ml-1">
            {t(I18nKey.COMMON$READ_MORE)}
          </Typography.Text>
        </p>
      </div>

      {/* Footer */}
      <div className="border-t border-[#525252] flex h-[54px] items-center justify-start px-4">
        <button
          type="button"
          onClick={onBuildClick}
          className="bg-white flex items-center justify-center h-[26px] px-2 rounded-[4px] w-[93px] hover:opacity-90 transition-opacity cursor-pointer"
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
