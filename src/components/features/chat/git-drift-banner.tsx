import { useTranslation } from "react-i18next";
import { RefreshCw, GitFork, X } from "lucide-react";

interface GitDriftBannerProps {
  summary: string;
  previousHead?: string | null;
  currentHead?: string | null;
  onRefreshBaseline: () => void;
  onDismiss: () => void;
}

/**
 * Non-blocking heads-up rendered when resuming a conversation whose
 * git workspace has drifted since the last completed agent turn.
 *
 * Tied to `useGitResumeDrift` — the hook compares HEAD SHA and working-
 * tree change count against a localStorage-backed baseline. The banner
 * shows when drift is detected and lets the user acknowledge by
 * refreshing the baseline (agent's own edits become the new ground
 * truth) or dismiss the warning for this turn.
 *
 * No automatic git operations are performed here — we never reset,
 * checkout, or stash anything. Just a heads-up.
 */
export function GitDriftBanner({
  summary,
  previousHead,
  currentHead,
  onRefreshBaseline,
  onDismiss,
}: GitDriftBannerProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="git-drift-banner"
      role="status"
      className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
    >
      <GitFork className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-amber-100">
          {t("GIT_DRIFT$BANNER_TITLE")}
        </div>
        <div className="mt-0.5 truncate opacity-90">{summary}</div>
        {(previousHead || currentHead) && (
          <div className="mt-0.5 font-mono text-[10px] text-amber-300/70">
            {previousHead && (
              <span>
                {t("GIT_DRIFT$PREV_HEAD", { hash: previousHead.slice(0, 8) })}
              </span>
            )}
            {previousHead && currentHead && <span className="mx-1">→</span>}
            {currentHead && (
              <span>
                {t("GIT_DRIFT$NOW_HEAD", { hash: currentHead.slice(0, 8) })}
              </span>
            )}
          </div>
        )}
        <div className="mt-1 flex gap-2">
          <button
            type="button"
            onClick={onRefreshBaseline}
            className="inline-flex items-center gap-1 rounded border border-amber-500/40 px-2 py-0.5 text-[11px] font-medium text-amber-100 transition-colors hover:bg-amber-500/20"
          >
            <RefreshCw className="h-3 w-3" aria-hidden />
            {t("GIT_DRIFT$ACKNOWLEDGE")}
          </button>
        </div>
      </div>
      <button
        type="button"
        aria-label={t("GIT_DRIFT$DISMISS_LABEL")}
        onClick={onDismiss}
        className="shrink-0 rounded p-0.5 text-amber-200/60 transition-colors hover:bg-amber-500/20 hover:text-amber-100"
      >
        <X className="h-3.5 w-3.5" aria-hidden />
      </button>
    </div>
  );
}
