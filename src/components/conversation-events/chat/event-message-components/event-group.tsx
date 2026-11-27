import type { SessionActivity } from "#/stores/session-activity-store";
import {
  sessionActivityTitle,
  sessionActivityLiveTitle,
} from "./session-activity";
import React from "react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { SuccessIndicator } from "#/components/features/chat/success-indicator";
import { getObservationResult } from "../event-content-helpers/get-observation-result";
import { formatEventTimestamp } from "#/utils/format-event-timestamp";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isACPToolCallEvent,
  isAgentErrorEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { I18nKey } from "#/i18n/declaration";
import {
  getEventGroupSummaryParts,
  getSingleToolEvent,
} from "./event-group-summary";
import { DisclosureCaret } from "#/components/shared/disclosure-caret";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { IsInEventGroupContext } from "../../../features/chat/is-in-event-group-context";
import { PathInteractiveContext } from "../../../features/chat/path-component";
import { buildActionById } from "../event-thought-helpers";

interface EventGroupProps {
  /** The events represented by this group. Used to compute the summary. */
  events: OpenHandsEvent[];
  session?: SessionActivity;
  hasReasoning?: boolean;
  /**
   * Full event history. Used to resolve the action behind each observation so
   * the summary counts it as the right kind of step. Falls back to `events`
   * when omitted.
   */
  allEvents?: OpenHandsEvent[];
  /** Reuse the message list's history index across every visible group. */
  actionById?: ReadonlyMap<string, ActionEvent>;
  /**
   * `true` while this group is the last item of the chat and the agent is
   * running. The header then shows live status instead of the summary.
   */
  isLive?: boolean;
  /**
   * Open state kept by the message list, so the group stays open when it
   * remounts. Falls back to local state when omitted.
   */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** The fully-rendered event messages to show when the group is expanded. */
  children: React.ReactNode;
}

/**
 * Collapsible container that wraps a run of consecutive agent action/observation
 * events into a single summary card.
 *
 * Collapsed, while live: the step that is running ("Ran the title tests"), or
 * "Thinking" between steps, with a spinner. Hook runs never take over the
 * title; they are quick and fire between steps.
 *
 * Collapsed otherwise: what the steps did, e.g. "Ran 3 commands, read a file".
 *
 * Expanded: one tool reveals its details directly; multiple tools keep their
 * individual disclosures in a bordered list.
 */
export function EventGroup({
  events,
  session,
  hasReasoning = false,
  allEvents,
  actionById,
  isLive = false,
  open,
  onOpenChange,
  children,
}: EventGroupProps) {
  const { t, i18n } = useTranslation("openhands");
  const [localExpanded, setLocalExpanded] = React.useState(false);
  const expanded = open ?? localExpanded;
  const toggleExpanded = () =>
    onOpenChange ? onOpenChange(!expanded) : setLocalExpanded(!expanded);
  const contentId = React.useId();
  const buttonId = `${contentId}-toggle`;
  const lookupSource = allEvents ?? events;
  const resolvedActions = React.useMemo(
    () => actionById ?? buildActionById(lookupSource),
    [actionById, lookupSource],
  );

  if (events.length === 0 && !session && !hasReasoning) {
    return null;
  }

  let summary: React.ReactNode = getEventGroupSummaryParts(
    events,
    lookupSource,
    resolvedActions,
  )
    .map(({ key, count }, index) => {
      const context = [
        count === 1 ? "single" : "",
        index > 0 ? "continuation" : "",
      ]
        .filter(Boolean)
        .join("_");
      return t(key, { count, context: context || undefined });
    })
    .join(t(I18nKey.EVENT_GROUP$SUMMARY_SEPARATOR));

  summary ||= t(
    session ? sessionActivityTitle(session) : I18nKey.THINKING$TITLE,
  );

  const singleTool = getSingleToolEvent(events);
  if (singleTool) {
    const actionId =
      isObservationEvent(singleTool) || isUserRejectObservation(singleTool)
        ? singleTool.action_id
        : undefined;
    const action = actionId ? resolvedActions.get(actionId) : undefined;
    summary = (
      <PathInteractiveContext.Provider value={false}>
        {getEventContent(singleTool, action).title}
      </PathInteractiveContext.Provider>
    );
  }

  if (isLive) {
    // An action still in the UI events is in flight, unless an AgentErrorEvent
    // resolved it (that event renders on its own and never replaces it).
    const erroredToolCallIds = new Set(
      lookupSource.filter(isAgentErrorEvent).map((e) => e.tool_call_id),
    );
    const running = events.find(
      (event) =>
        (isActionEvent(event) && !erroredToolCallIds.has(event.tool_call_id)) ||
        (isACPToolCallEvent(event) &&
          (event.status === "pending" || event.status === "in_progress")),
    );
    summary = running ? (
      <PathInteractiveContext.Provider value={false}>
        {getEventContent(running).title}
      </PathInteractiveContext.Provider>
    ) : (
      t(I18nKey.ACTION_MESSAGE$THINK)
    );
  }

  if (session && session.phase !== "complete")
    summary = t(sessionActivityLiveTitle(session));
  const timestampLabel = formatEventTimestamp(
    singleTool?.timestamp,
    i18n?.language,
  );
  const singleResult =
    singleTool && isObservationEvent(singleTool)
      ? getObservationResult(singleTool)
      : undefined;
  const busy =
    session?.phase === "pending" ||
    session?.phase === "connecting" ||
    (session?.phase !== "failed" && isLive);

  return (
    // Agent text carries a large top margin for paragraph spacing; trim it
    // when the text follows a group so the group sits evenly between them.
    <div
      className="mt-2 w-full py-1 text-sm [&+article]:mt-2"
      data-testid="event-group"
    >
      <span
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {busy ? summary : null}
      </span>
      <StyledTooltip
        content={<time dateTime={singleTool?.timestamp}>{timestampLabel}</time>}
        placement="top"
        isDisabled={!timestampLabel}
      >
        <button
          id={buttonId}
          type="button"
          onClick={toggleExpanded}
          aria-controls={contentId}
          aria-expanded={expanded}
          aria-label={
            expanded
              ? t(I18nKey.EVENT_GROUP$COLLAPSE)
              : t(I18nKey.EVENT_GROUP$EXPAND)
          }
          data-testid="event-group-toggle"
          className="flex max-w-full items-center gap-2 text-left text-muted cursor-pointer"
        >
          <span className="truncate">{summary}</span>
          <DisclosureCaret open={expanded} />
          {singleResult === "timeout" && (
            <SuccessIndicator status={singleResult} />
          )}
          {singleTool && isUserRejectObservation(singleTool) && (
            <span
              data-testid="rejected-indicator"
              className="shrink-0 text-danger"
            >
              {t(I18nKey.EVENT$REJECTED)}
            </span>
          )}
          {busy && (
            <LoaderCircle
              data-testid="spinner-icon"
              className="h-3.5 w-3.5 flex-shrink-0 animate-spin text-muted motion-reduce:animate-none"
            />
          )}
        </button>
      </StyledTooltip>

      {expanded && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={buttonId}
          className="mt-1.5 flex flex-col divide-y divide-border-subtle/50 rounded-lg border border-border-subtle/50 [&>*]:px-3"
          data-testid="event-group-content"
        >
          <IsInEventGroupContext.Provider value>
            {children}
          </IsInEventGroupContext.Provider>
        </div>
      )}
    </div>
  );
}
