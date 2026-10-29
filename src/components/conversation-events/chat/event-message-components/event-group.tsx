import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
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
import { DisclosureCaret } from "#/components/shared/disclosure-caret";
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
 *   the title of the action in flight on the left, with the
 *   "{completed}/{total} actions completed" progress and a spinner on the right.
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
        className={
          showProgress
            ? "w-full flex items-center justify-between gap-2 text-left text-muted cursor-pointer"
            : "flex max-w-full items-center gap-2 text-left text-muted cursor-pointer"
        }
      >
        {showProgress ? (
          // Live step on the left, progress and spinner pinned on the right.
          <>
            <span className="flex items-center gap-2 min-w-0">
              <span className="truncate">
                <PathInteractiveContext.Provider value={false}>
                  {latestTitle ?? summary}
                </PathInteractiveContext.Provider>
              </span>
              <DisclosureCaret open={expanded} />
            </span>
            <span className="flex items-center flex-shrink-0">
              <span className="truncate">{summary}</span>
              <LoaderCircle
                data-testid="spinner-icon"
                className="h-4 w-4 ml-2 inline animate-spin text-muted"
              />
            </span>
          </>
        ) : (
          <>
            <span className="truncate">{summary}</span>
            <DisclosureCaret open={expanded} />
          </>
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
