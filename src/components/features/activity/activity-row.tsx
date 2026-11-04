import { useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  getActivityMetrics,
  getActivityStatusDescriptor,
  resolveDescriptorText,
  type ActivityStatusKind,
  type SubagentActivity,
} from "./activity-view-model";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { EventTitleDescriptor } from "#/components/conversation-events/chat/event-content-helpers/get-action-event-title";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { formatTimeDelta } from "#/utils/format-time-delta";

interface ActivityRowProps {
  conversation: AppConversation;
  latestActivity: EventTitleDescriptor | null;
  subagents: SubagentActivity[];
}

const STATUS_DOT_CLASS: Record<ActivityStatusKind, string> = {
  running: "bg-status-success",
  waiting: "bg-warning",
  paused: "bg-muted",
  error: "bg-status-error",
  finished: "bg-status-success",
  idle: "bg-muted",
  unknown: "bg-muted",
};

// i18next resolves these base keys to the `_one`/`_other` plural forms from
// `count`; the plural suffix is not part of the `I18nKey` enum, so the base
// key is kept as a constant (mirrors HOOK_COUNT_I18N_KEY).
const SUBAGENT_COUNT_I18N_KEY = "ACTIVITY$SUBAGENT_COUNT";
const TOKENS_I18N_KEY = "ACTIVITY$TOKENS";

/**
 * One row of the live activity view: the agent's status, what it is doing
 * right now, its subagent fan-out, and the cost/tokens it has spent so far.
 * The whole row links into the conversation so a click drills down; it is
 * read-only and carries no conversation mutations.
 */
// @spec LAV-002 — A row conveys status, current step, and spend
// @spec LAV-003 — Subagent fan-out is derived from the event stream
export function ActivityRow({
  conversation,
  latestActivity,
  subagents,
}: ActivityRowProps) {
  const { t } = useTranslation("openhands");
  const backendScopedPath = useBackendScopedPath();
  const status = getActivityStatusDescriptor(conversation.execution_status);
  const metrics = getActivityMetrics(conversation.metrics);

  const activityText = latestActivity
    ? resolveDescriptorText(latestActivity)
    : t(I18nKey.ACTIVITY$NO_ACTIVITY);

  return (
    <NavigationLink
      to={backendScopedPath(`/conversations/${conversation.id}`)}
      data-testid="activity-row"
      aria-label={conversation.title ?? conversation.id}
      className={cn(
        "flex flex-col gap-2 rounded-lg border bg-surface p-3 transition-colors",
        "hover:bg-surface-raised",
        status.needsAttention ? "border-warning" : "border-border",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "h-2 w-2 shrink-0 rounded-full",
            STATUS_DOT_CLASS[status.kind],
          )}
          aria-hidden
        />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-content">
          {conversation.title ?? conversation.id}
        </span>
        <span
          data-testid="activity-status"
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium",
            status.needsAttention
              ? "bg-warning/15 text-warning"
              : "bg-surface-raised text-muted",
          )}
        >
          {t(status.labelKey)}
        </span>
      </div>

      <div className="flex items-center gap-2 text-xs text-muted">
        <span className="min-w-0 flex-1 truncate" title={activityText}>
          {activityText}
        </span>
        <span className="shrink-0">
          {formatTimeDelta(conversation.updated_at)}{" "}
          {t(I18nKey.CONVERSATION$AGO)}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {status.needsAttention && (
          <span
            data-testid="activity-attention"
            className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 font-medium text-warning"
          >
            <AlertTriangle className="size-3" aria-hidden />
            {t(I18nKey.ACTIVITY$NEEDS_ATTENTION)}
          </span>
        )}

        {subagents.length > 0 && (
          <span
            data-testid="activity-subagents"
            className="inline-flex items-center gap-1 rounded-full bg-surface-raised px-2 py-0.5 text-muted"
          >
            {t(SUBAGENT_COUNT_I18N_KEY, { count: subagents.length })}
          </span>
        )}

        <span
          data-testid="activity-metrics"
          className="ml-auto inline-flex items-center gap-2 text-muted"
        >
          <span>
            {metrics.cost === null
              ? t(I18nKey.ACTIVITY$METRICS_UNKNOWN)
              : `$${metrics.cost.toFixed(4)}`}
          </span>
          <span>
            {metrics.totalTokens === null
              ? t(I18nKey.ACTIVITY$METRICS_UNKNOWN)
              : t(TOKENS_I18N_KEY, {
                  count: metrics.totalTokens,
                })}
          </span>
        </span>
      </div>
    </NavigationLink>
  );
}
