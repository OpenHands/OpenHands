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
  const { t } = useTranslation("openhands");

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
      <div className="w-full h-1.5 bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-foreground transition-all duration-300"
          // runtime usage-percentage width
          style={{ width: `${progressWidth}%` }}
        />
      </div>
      <div className="flex justify-end">
        <span className="text-xs text-[var(--oh-muted)]">
          {perTurnToken.toLocaleString()} / {contextWindow.toLocaleString()} (
          {usagePercentage.toFixed(2)}% {t(I18nKey.CONVERSATION$USED)})
        </span>
      </div>
    </div>
  );
}
