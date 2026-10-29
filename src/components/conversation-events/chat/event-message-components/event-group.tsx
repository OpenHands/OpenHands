import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { OpenHandsEvent, ActionEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isAgentErrorEvent,
  isHookExecutionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { I18nKey } from "#/i18n/declaration";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { getEventGroupSummaryParts } from "./event-group-summary";
import { IsInEventGroupContext } from "../../../features/chat/is-in-event-group-context";
import { PathInteractiveContext } from "../../../features/chat/path-component";

interface EventGroupProps {
  /** The events represented by this group. Used to compute the summary. */
  events: OpenHandsEvent[];
  /**
   * Full event history. Used to resolve the action that produced the latest
   * observation in the group so the summary title matches what the individual
   * card would show (e.g. "Editing path/to/file"), and to find the
   * `AgentErrorEvent` that resolved an action without an observation. Falls
   * back to `events` when omitted.
   */
  allEvents?: OpenHandsEvent[];
  /**
   * `true` once an event outside this group has been emitted after it, so the
   * group is no longer the "live" tail of the chat. A group whose actions are
   * still running shows its latest action and progress only while it is the
   * live tail; otherwise it shows what its steps did ("Ran 3 commands").
   */
  isFinalized?: boolean;
  /** The fully-rendered event messages to show when the group is expanded. */
  children: React.ReactNode;
}

/**
 * Collapsible container that wraps a run of consecutive agent action/observation
 * events into a single summary card.
 *
 * Collapsed, while an action is still running in the live tail of the chat:
 *   one line with the title of the action in flight, the
 *   "{completed}/{total} actions completed" progress and a spinner.
 *
 * Collapsed otherwise: what the steps did, e.g. "Ran 3 commands · Read 1 file".
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
  // Progress counts tool steps; folded-in hook runs are not actions.
  const totalCount = events.filter((e) => !isHookExecutionEvent(e)).length;
  const completedCount = totalCount - pendingActions.length;
  const isRunning = pendingActions.length > 0;

  // Title of the most recent groupable event. While running this is the
  // pending action; otherwise it's the latest observation (or rejection), with
  // its originating action looked up so the title can be the action-style
  // summary ("Editing path/to/file") instead of the observation default.
  const latestEvent = events[events.length - 1];
  let latestTitle: React.ReactNode = null;
  if (latestEvent) {
    if (isActionEvent(latestEvent)) {
      latestTitle = getEventContent(latestEvent).title;
    } else if (
      isObservationEvent(latestEvent) ||
      isUserRejectObservation(latestEvent)
    ) {
      const correspondingAction = lookupSource.find(
        (e): e is ActionEvent =>
          isActionEvent(e) && e.id === latestEvent.action_id,
      );
      latestTitle = getEventContent(latestEvent, correspondingAction).title;
    }
  }

  const showProgress = isRunning && !isFinalized;
  const summary = showProgress
    ? t(I18nKey.EVENT_GROUP$ACTIONS_PROGRESS, {
        completed: completedCount,
        total: totalCount,
      })
    : getEventGroupSummaryParts(events, lookupSource)
        .map(({ key, count }) => t(key, { count }))
        .join(" · ");

  const Chevron = expanded ? ArrowUp : ArrowDown;

  return (
    <div className="my-1 w-full py-1 text-sm" data-testid="event-group">
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
        {showProgress && latestTitle ? (
          <span className="truncate">
            <PathInteractiveContext.Provider value={false}>
              {latestTitle}
            </PathInteractiveContext.Provider>
          </span>
        ) : null}
        <span
          className={showProgress && latestTitle ? "flex-shrink-0" : "truncate"}
        >
          {showProgress && latestTitle ? `· ${summary}` : summary}
        </span>
        {showProgress ? (
          <LoaderCircle
            data-testid="spinner-icon"
            className="h-4 w-4 inline flex-shrink-0 animate-spin text-muted"
          />
        ) : null}
        <Chevron className="h-4 w-4 fill-muted flex-shrink-0" />
      </button>

      {expanded && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={buttonId}
          className="mt-1.5 flex flex-col divide-y divide-border-subtle rounded-lg border border-border-subtle [&>*]:px-3"
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
