import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isAgentErrorEvent,
  isHookExecutionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { I18nKey } from "#/i18n/declaration";
import { getEventGroupSummaryParts } from "./event-group-summary";
import { DisclosureCaret } from "#/components/shared/disclosure-caret";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { IsInEventGroupContext } from "../../../features/chat/is-in-event-group-context";
import { PathInteractiveContext } from "../../../features/chat/path-component";

interface EventGroupProps {
  /** The events represented by this group. Used to compute the summary. */
  events: OpenHandsEvent[];
  /**
   * Full event history. Used to resolve the action behind each observation so
   * the summary counts it as the right kind of step, and to find the
   * `AgentErrorEvent` that resolved an action without an observation. Falls
   * back to `events` when omitted.
   */
  allEvents?: OpenHandsEvent[];
  /**
   * `true` once an event outside this group has been emitted after it, so the
   * group is no longer the "live" tail of the chat. The live tail shows its
   * latest step and a spinner while an action runs; a finalized group shows
   * what its steps did ("Ran 3 commands").
   */
  isFinalized?: boolean;
  /** The fully-rendered event messages to show when the group is expanded. */
  children: React.ReactNode;
}

/**
 * Collapsible container that wraps a run of consecutive agent action/observation
 * events into a single summary card.
 *
 * Collapsed, while the group is the live tail of the chat: the latest tool
 * step's title ("Ran the title tests"), with a spinner while an action runs.
 * Hook runs don't take over the title; they are quick and fire between steps.
 *
 * Collapsed once the agent has moved on: what the steps did, e.g.
 * "Ran 3 commands · Read 1 file". The layout is the same in both states.
 *
 * Expanded: the children in a bordered list, so each individual
 * action/observation can still be expanded the way it was before grouping.
 */
export function EventGroup({
  events,
  allEvents,
  isFinalized = false,
  children,
}: EventGroupProps) {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = React.useState(false);
  const contentId = React.useId();
  const buttonId = `${contentId}-toggle`;

  if (events.length === 0) {
    return null;
  }

  // Each ObservationEvent (or UserRejectObservation) in the group is a
  // resolved action. An ActionEvent that's still here (i.e. not yet replaced
  // in the UI events array) is an action currently in flight, unless an
  // AgentErrorEvent resolved it: that event carries no action id and renders
  // on its own, so it never replaces the action it answers.
  const lookupSource = allEvents ?? events;
  const unreplacedActions = events.filter(isActionEvent);
  const erroredToolCallIds =
    unreplacedActions.length > 0
      ? new Set(
          lookupSource.filter(isAgentErrorEvent).map((e) => e.tool_call_id),
        )
      : null;
  const pendingActions = unreplacedActions.filter(
    (action) => !erroredToolCallIds?.has(action.tool_call_id),
  );
  const showSpinner = pendingActions.length > 0 && !isFinalized;
  const latestStep = isFinalized
    ? undefined
    : [...events].reverse().find((e) => !isHookExecutionEvent(e));
  let latestTitle: React.ReactNode = null;
  if (latestStep && isActionEvent(latestStep)) {
    latestTitle = getEventContent(latestStep).title;
  } else if (
    latestStep &&
    (isObservationEvent(latestStep) || isUserRejectObservation(latestStep))
  ) {
    const action = lookupSource.find(
      (e): e is ActionEvent =>
        isActionEvent(e) && e.id === latestStep.action_id,
    );
    latestTitle = getEventContent(latestStep, action).title;
  }
  const summary = latestTitle ? (
    <PathInteractiveContext.Provider value={false}>
      {latestTitle}
    </PathInteractiveContext.Provider>
  ) : (
    getEventGroupSummaryParts(events, lookupSource)
      .map(({ key, count }) => t(key, { count }))
      .join(" · ")
  );

  return (
    // Agent text carries a large top margin for paragraph spacing; trim it
    // when the text follows a group so the group sits evenly between them.
    <div
      className="mt-2 w-full py-1 text-sm [&+article]:mt-2"
      data-testid="event-group"
    >
      <button
        id={buttonId}
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
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
        {showSpinner && (
          <LoaderCircle
            data-testid="spinner-icon"
            className="h-3.5 w-3.5 flex-shrink-0 animate-spin text-muted"
          />
        )}
      </button>

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
