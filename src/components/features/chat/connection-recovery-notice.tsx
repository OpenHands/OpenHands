import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { ConnectionRecoveryState } from "#/hooks/use-connection-error-grace-period";

export function ConnectionRecoveryNotice({
  state,
  onRetry,
}: {
  state: ConnectionRecoveryState;
  onRetry: () => void;
}) {
  const { t } = useTranslation("openhands");
  if (state === "idle") return null;
  return (
    <div
      role="status"
      className="relative z-10 flex items-center justify-between gap-3 rounded-lg border border-border bg-surface-raised p-2 text-sm text-muted"
    >
      <span>
        {t(
          state === "retryable"
            ? I18nKey.CHAT_INTERFACE$STILL_RECONNECTING
            : I18nKey.CHAT_INTERFACE$RECONNECTING,
        )}
      </span>
      {state === "retryable" && (
        <button
          type="button"
          onClick={onRetry}
          className="shrink-0 rounded px-2 py-1 text-foreground hover:bg-white/5"
        >
          {t(I18nKey.CHAT_INTERFACE$MESSAGE_RETRY)}
        </button>
      )}
    </div>
  );
}
