import { useId } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useTokenUsageBreakdown } from "#/hooks/use-token-usage-breakdown";
import type {
  TokenUsageActivity,
  TokenUsageActivityRow,
} from "#/utils/token-usage-breakdown";
import { cn } from "#/utils/utils";

const TOOL_NAME_SEPARATOR = " + ";

function ActivityLabel({ activity }: { activity: TokenUsageActivity }) {
  const { t } = useTranslation("openhands");

  switch (activity.kind) {
    case "tools": {
      // Tool names are the identifiers the agent called; they stay as-is.
      const toolNames = activity.toolNames.join(TOOL_NAME_SEPARATOR);
      return (
        <span className="min-w-0 truncate font-mono" title={toolNames}>
          {toolNames}
        </span>
      );
    }
    case "agent_message":
      return (
        <span>{t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_AGENT_MESSAGES)}</span>
      );
    case "condensation":
      return <span>{t(I18nKey.ACTION_MESSAGE$CONDENSATION)}</span>;
    case "unattributed":
      return <span>{t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_UNATTRIBUTED)}</span>;
    case "other_llm":
      return (
        <span className="min-w-0 truncate">
          {t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_OTHER_LLM, {
            usageId: activity.usageId,
          })}
        </span>
      );
    default:
      return null;
  }
}

interface ActivityBarProps {
  row: TokenUsageActivityRow;
  maxTokens: number;
  totalTokens: number;
}

function ActivityBar({ row, maxTokens, totalTokens }: ActivityBarProps) {
  const { t } = useTranslation("openhands");
  const widthPercent = maxTokens > 0 ? (row.totalTokens / maxTokens) * 100 : 0;
  const share = totalTokens > 0 ? row.totalTokens / totalTokens : 0;

  return (
    <li
      data-testid="token-usage-activity"
      data-activity-key={row.key}
      className="flex flex-col gap-1"
    >
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <ActivityLabel activity={row.activity} />
        <span className="shrink-0 tabular-nums font-semibold">
          {row.totalTokens.toLocaleString()}
          <span className="ml-1.5 text-xs font-normal text-muted">
            {share.toLocaleString(undefined, {
              style: "percent",
              maximumFractionDigits: 1,
            })}
          </span>
        </span>
      </div>
      {/* Bars share one baseline and scale to the largest activity. */}
      <div className="h-2 w-full" aria-hidden>
        <div
          className={cn(
            "h-full rounded-r-sm bg-foreground",
            row.totalTokens > 0 && "min-w-0.5",
          )}
          // runtime width relative to the largest activity
          style={{ width: `${widthPercent}%` }}
        />
      </div>
      <span className="text-xs text-muted tabular-nums">
        {t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_DETAILS, {
          calls: row.calls.toLocaleString(),
          input: row.inputTokens.toLocaleString(),
          output: row.outputTokens.toLocaleString(),
        })}
      </span>
    </li>
  );
}

/**
 * Usage-tab card that ranks the conversation's activities by the tokens their
 * LLM calls used. Renders nothing until the agent server reports a call.
 */
export function TokenUsageBreakdown() {
  const { t } = useTranslation("openhands");
  const titleId = useId();
  const { breakdown, isLoadingHistory, isHistoryError } =
    useTokenUsageBreakdown();

  if (!breakdown || breakdown.rows.length === 0) {
    return null;
  }

  const maxTokens = breakdown.rows[0].totalTokens;

  return (
    <section
      data-testid="token-usage-breakdown"
      aria-labelledby={titleId}
      className="rounded-md border border-border bg-surface-raised p-3"
    >
      <div className="grid gap-3">
        <div className="flex flex-col gap-1">
          <h3 id={titleId} className="text-lg font-semibold">
            {t(I18nKey.CONVERSATION$TOKEN_USAGE_BY_ACTIVITY)}
          </h3>
          <span className="text-xs text-muted">
            {t(I18nKey.CONVERSATION$TOKEN_USAGE_BY_ACTIVITY_DESCRIPTION)}
          </span>
        </div>
        {isLoadingHistory && (
          <span role="status" className="text-xs text-muted">
            {t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_LOADING_HISTORY)}
          </span>
        )}
        {isHistoryError && (
          <span role="alert" className="text-xs text-muted">
            {t(I18nKey.CONVERSATION$TOKEN_ACTIVITY_HISTORY_ERROR)}
          </span>
        )}
        <ul className="grid gap-3">
          {breakdown.rows.map((row) => (
            <ActivityBar
              key={row.key}
              row={row}
              maxTokens={maxTokens}
              totalTokens={breakdown.totalTokens}
            />
          ))}
        </ul>
      </div>
    </section>
  );
}
