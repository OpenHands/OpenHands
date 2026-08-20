import { Trans } from "react-i18next";
import type { OHEvent } from "#/stores/use-event-store";
import type { UserRejectObservation } from "#/types/agent-server/core";
import {
  isACPToolCallEvent,
  isActionEvent,
  isAgentErrorEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import {
  getACPToolCallTitleKey,
  stripRedundantTitlePrefix,
} from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";
import {
  getActionEventTitleDescriptor,
  getActionSummaryTitle,
  type EventTitleDescriptor,
} from "#/components/conversation-events/chat/event-content-helpers/get-action-event-title";
import { MonoComponent } from "./mono-component";
import { PathComponent } from "./path-component";

const THINKING_ACTIVITY: EventTitleDescriptor = {
  kind: "translation",
  key: "ACTION_MESSAGE$THINK",
  values: {},
};

const LIVE_ACTION_KINDS = new Set([
  "ExecuteBashAction",
  "TerminalAction",
  "FileEditorAction",
  "StrReplaceEditorAction",
  "MCPToolAction",
  "InvokeSkillAction",
  "TaskAction",
  "ThinkAction",
  "TaskTrackerAction",
  "GrepAction",
  "GlobAction",
  "BrowserNavigateAction",
  "BrowserClickAction",
  "BrowserTypeAction",
  "BrowserGetStateAction",
  "BrowserGetContentAction",
  "BrowserScrollAction",
  "BrowserGoBackAction",
  "BrowserListTabsAction",
  "BrowserSwitchTabAction",
  "BrowserCloseTabAction",
]);

const getLiveActionTitle = (event: OHEvent): EventTitleDescriptor | null => {
  if (!isActionEvent(event)) return null;

  const summary = getActionSummaryTitle(event);
  if (summary) {
    return { kind: "text", text: summary };
  }

  return LIVE_ACTION_KINDS.has(event.action.kind)
    ? getActionEventTitleDescriptor(event)
    : null;
};

const isUserRejectObservation = (
  event: OHEvent,
): event is UserRejectObservation =>
  event.source === "environment" && "rejection_reason" in event;

export const deriveLiveActivity = (
  events: readonly OHEvent[],
): EventTitleDescriptor => {
  const resolvedActionIds = new Set<string>();
  const resolvedToolCallIds = new Set<string>();

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.isFromPlanningAgent) {
      continue;
    }

    if (isObservationEvent(event) || isUserRejectObservation(event)) {
      resolvedActionIds.add(event.action_id);
      resolvedToolCallIds.add(event.tool_call_id);
      continue;
    }

    if (isAgentErrorEvent(event)) {
      resolvedToolCallIds.add(event.tool_call_id);
      continue;
    }

    if (isACPToolCallEvent(event)) {
      const isInProgress =
        event.status === "pending" || event.status === "in_progress";
      if (!isInProgress) {
        resolvedToolCallIds.add(event.tool_call_id);
        continue;
      }
      if (resolvedToolCallIds.has(event.tool_call_id)) {
        continue;
      }

      const title = stripRedundantTitlePrefix(event);
      return title
        ? {
            kind: "translation",
            key: getACPToolCallTitleKey(event),
            values: { title },
          }
        : THINKING_ACTIVITY;
    }

    if (
      isActionEvent(event) &&
      !resolvedActionIds.has(event.id) &&
      !resolvedToolCallIds.has(event.tool_call_id)
    ) {
      return getLiveActionTitle(event) ?? THINKING_ACTIVITY;
    }
  }

  return THINKING_ACTIVITY;
};

interface TypingIndicatorProps {
  readonly events: readonly OHEvent[];
}

export function TypingIndicator({ events }: TypingIndicatorProps) {
  const activity = deriveLiveActivity(events);

  return (
    <div
      className="flex min-w-0 max-w-full items-center gap-2 rounded-full border border-[var(--oh-border)] bg-[var(--oh-surface)] px-3 py-1.5 text-xs text-[var(--oh-text-secondary)]"
      data-testid="live-activity-chip"
      role="status"
      aria-live="polite"
    >
      <span
        aria-hidden="true"
        className="size-1.5 shrink-0 animate-pulse rounded-full bg-[var(--oh-status-success)] motion-reduce:animate-none"
      />
      <span className="min-w-0 truncate">
        {activity.kind === "text" ? (
          activity.text
        ) : (
          <Trans
            ns="openhands"
            i18nKey={activity.key}
            values={activity.values}
            components={{
              path: <PathComponent />,
              cmd: <MonoComponent />,
            }}
          />
        )}
      </span>
    </div>
  );
}
