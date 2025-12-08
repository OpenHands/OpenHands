import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface ContextWindowSectionProps {
  perTurnToken: number;
  contextWindow: number;
}

export function ContextWindowSection({
  perTurnToken,
  contextWindow,
}: ContextWindowSectionProps) {
  const { t } = useTranslation();

  const usagePercentage =
    contextWindow > 0 ? (perTurnToken / contextWindow) * 100 : 0;
  const progressWidth = Math.min(100, usagePercentage);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$CONTEXT_WINDOW)}
        </span>
      </div>
      <div className="w-full h-1.5 bg-neutral-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${progressWidth}%` }}
        />
      </div>
      <div className="flex justify-end">
        <span className="text-xs text-neutral-400">
          {perTurnToken.toLocaleString()} / {contextWindow.toLocaleString()} (
          {usagePercentage.toFixed(2)}% {t(I18nKey.CONVERSATION$USED)})
        </span>
      </div>
    </div>
  );
}
