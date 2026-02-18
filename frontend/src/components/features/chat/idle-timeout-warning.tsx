import { useTranslation } from "react-i18next";
import { useIdleTimeout } from "#/hooks/query/use-idle-timeout";
import { I18nKey } from "#/i18n/declaration";

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

export function IdleTimeoutWarning() {
  const { t } = useTranslation();
  const { isWarning, remainingSeconds, isEnabled } = useIdleTimeout();

  if (!isEnabled || !isWarning) return null;

  return (
    <div
      className="w-full rounded-lg p-2 border border-amber-500 bg-amber-900/60 flex gap-2 items-center text-white text-sm"
      data-testid="idle-timeout-warning"
    >
      <span>
        {t(I18nKey.SANDBOX$IDLE_TIMEOUT_WARNING, {
          time: formatTime(remainingSeconds),
        })}
      </span>
    </div>
  );
}
