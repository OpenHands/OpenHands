import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { RefreshCw, Check, TriangleAlert } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";

export type GitSyncActivityState = "idle" | "running" | "succeeded" | "failed";

interface GitSyncActivityRowProps {
  state: GitSyncActivityState;
  /** When the running cycle started, for the elapsed-time hint. */
  startedAt: string | null;
  /** Automations still waiting to be pushed, shown while a cycle runs. */
  pendingCount: number;
}

/**
 * Re-render every second while a cycle runs. The elapsed hint is derived from
 * the clock, not from props, and the status poll returns identical JSON for as
 * long as the cycle is in flight -- react-query's structural sharing then
 * hands back the same object, nothing re-renders, and the hint would sit at
 * whatever it read when the cycle started.
 */
function useSecondsTick(active: boolean) {
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(() => setTick((count) => count + 1), 1_000);
    return () => clearInterval(timer);
  }, [active]);
}

/**
 * The page's own account of the sync cycle, in place of the fire-and-forget
 * success toast: the trigger returns as soon as the cycle is scheduled, so a
 * toast said "started" and then never came back with an outcome.
 */
export function GitSyncActivityRow({
  state,
  startedAt,
  pendingCount,
}: GitSyncActivityRowProps) {
  const { t } = useTranslation("openhands");
  const isRunning = state === "running";

  useSecondsTick(isRunning && startedAt !== null);

  if (state === "idle") return null;

  const isFailed = state === "failed";

  return (
    <div
      data-testid="git-sync-activity-row"
      data-state={state}
      className={cn(
        "mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md px-3 py-2 text-sm",
        isRunning && "bg-tertiary text-content",
        state === "succeeded" && "bg-green-500/10 text-green-300",
        isFailed && "bg-red-500/10 text-red-300",
      )}
    >
      {isRunning && <RefreshCw className="size-4 animate-spin" aria-hidden />}
      {state === "succeeded" && <Check className="size-4" aria-hidden />}
      {isFailed && <TriangleAlert className="size-4" aria-hidden />}

      <span className="font-medium">
        {isRunning && t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNCING)}
        {state === "succeeded" &&
          t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_COMPLETED)}
        {isFailed && t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_FAILED)}
      </span>

      {isRunning && startedAt && (
        <span className="text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_STARTED_AGO, {
            time: formatTimeDelta(startedAt),
          })}
        </span>
      )}
      {isRunning && pendingCount > 0 && (
        <span className="text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_PENDING_COUNT, {
            count: pendingCount,
          })}
        </span>
      )}
    </div>
  );
}
