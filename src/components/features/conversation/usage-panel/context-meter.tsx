import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

/** Context fill percentage above which the meter warns and the compact CTA becomes prominent. */
export const CONTEXT_FILL_WARNING_PERCENT = 70;
/** Context fill percentage above which the meter signals danger. */
export const CONTEXT_FILL_DANGER_PERCENT = 90;

export type ContextFillTone = "neutral" | "warning" | "danger";

export function getContextFillPercent(
  perTurnToken: number,
  contextWindow: number,
): number {
  return contextWindow > 0 ? (perTurnToken / contextWindow) * 100 : 0;
}

export function getContextFillTone(usagePercentage: number): ContextFillTone {
  if (usagePercentage > CONTEXT_FILL_DANGER_PERCENT) {
    return "danger";
  }
  if (usagePercentage > CONTEXT_FILL_WARNING_PERCENT) {
    return "warning";
  }
  return "neutral";
}

interface ContextMeterProps {
  /** Tokens currently held in the agent's context (`per_turn_token`). */
  perTurnToken: number;
  /** Model context window size in tokens. */
  contextWindow: number;
}

/**
 * Context-fill progress bar: neutral below
 * {@link CONTEXT_FILL_WARNING_PERCENT}, amber above it, red above
 * {@link CONTEXT_FILL_DANGER_PERCENT}. Shows the raw token numbers underneath.
 */
export function ContextMeter({
  perTurnToken,
  contextWindow,
}: ContextMeterProps) {
  const { t } = useTranslation("openhands");

  // Some models (e.g. via OpenRouter) report no context window size; show the
  // raw token count without a misleading "x / 0" or "0.0% used" readout.
  const isWindowUnknown = contextWindow <= 0;

  const usagePercentage = getContextFillPercent(perTurnToken, contextWindow);
  const progressWidth = Math.min(100, usagePercentage);
  const tone = getContextFillTone(usagePercentage);
  const isWarning = tone === "warning";
  const isDanger = tone === "danger";
  const roundedPercentage = Math.round(usagePercentage);
  const remainingPercentage = Math.max(0, 100 - roundedPercentage);
  const usagePercentLabel = isWindowUnknown
    ? t(I18nKey.CONVERSATION$CONTEXT_WINDOW_UNKNOWN)
    : `${roundedPercentage}% ${t(I18nKey.CONVERSATION$USED)} (${remainingPercentage}% ${t(I18nKey.CONVERSATION$LEFT)})`;

  return (
    <div data-testid="context-meter" className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$CONTEXT_WINDOW)}
        </span>
        <span
          className={cn(
            "shrink-0 text-xs",
            isDanger
              ? "text-red-500"
              : isWarning
                ? "text-amber-500"
                : "text-[var(--oh-muted)]",
          )}
        >
          {usagePercentLabel}
        </span>
      </div>
      <div className="relative h-1.5 w-full rounded-full bg-tertiary">
        <div
          data-testid="context-meter-bar"
          className={cn(
            "absolute inset-y-0 left-0 rounded-full transition-all duration-300",
            isDanger
              ? "bg-red-500"
              : isWarning
                ? "bg-amber-500"
                : "bg-foreground",
          )}
          // runtime usage-percentage width
          style={{ width: `${progressWidth}%` }}
        />
      </div>
      <div className="flex justify-end">
        <span className="text-xs text-[var(--oh-muted)]">
          {isWindowUnknown
            ? perTurnToken.toLocaleString()
            : `${perTurnToken.toLocaleString()} / ${contextWindow.toLocaleString()}`}
        </span>
      </div>
    </div>
  );
}
