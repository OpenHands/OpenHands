import { useTranslation } from "react-i18next";
import { AlertTriangle, X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useSandboxOOM } from "#/hooks/use-sandbox-oom";

export interface SandboxOOMRecoveryBannerProps {
  conversationId: string | null | undefined;
}

/**
 * Notice shown when a sandbox process was terminated by the Linux OOM-killer
 * (exit code 137). Even after the sandbox automatically recovers or reconnects,
 * this banner explains why previous tasks halted and provides actionable advice.
 */
export function SandboxOOMRecoveryBanner({
  conversationId,
}: SandboxOOMRecoveryBannerProps) {
  const { t } = useTranslation("openhands");
  const { activeOOM, dismissActiveOOM } = useSandboxOOM(conversationId);

  if (!activeOOM) {
    return null;
  }

  return (
    <div
      role="status"
      data-testid="sandbox-oom-recovery-banner"
      className="mb-2 flex w-full flex-col gap-2 rounded-lg border border-warning/40 bg-warning-soft p-3 text-warning-soft-foreground dark:bg-warning/10 dark:text-warning font-sans"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <AlertTriangle
            className="size-4 shrink-0 text-warning"
            aria-hidden="true"
          />
          <span className="text-xs font-semibold text-foreground">
            {t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_TITLE)}
          </span>
        </div>
        <button
          type="button"
          data-testid="sandbox-oom-dismiss-btn"
          onClick={dismissActiveOOM}
          className="rounded p-1 text-muted hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10 cursor-pointer"
          aria-label={t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_DISMISS)}
        >
          <X className="size-3.5" />
        </button>
      </div>

      <p className="text-xs text-muted leading-relaxed">
        {t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_DESCRIPTION)}
      </p>

      <ul className="list-disc pl-4 text-xs text-muted space-y-0.5">
        <li>{t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_ADVICE_CHUNKS)}</li>
        <li>{t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_ADVICE_MEMORY)}</li>
      </ul>

      <div className="mt-1 flex justify-end">
        <button
          type="button"
          data-testid="sandbox-oom-bottom-dismiss-btn"
          onClick={dismissActiveOOM}
          className="cursor-pointer rounded-md border border-border px-2.5 py-1 text-xs font-medium text-foreground hover:bg-interactive-hover"
        >
          {t(I18nKey.CHAT_INTERFACE$SANDBOX_OOM_DISMISS)}
        </button>
      </div>
    </div>
  );
}
