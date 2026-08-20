import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { OpenHandsEvent, ActionEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import { I18nKey } from "#/i18n/declaration";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { IsInEventGroupContext } from "../../../features/chat/is-in-event-group-context";

interface EventGroupProps {
  /** The events represented by this group. Used to compute the summary. */
  events: OpenHandsEvent[];
  /**
   * Full event history. Used to resolve the action that produced the latest
   * observation in the group so the summary title matches what the individual
   * card would show (e.g. "Editing path/to/file"). Falls back to `events` when
   * omitted.
   */
  allEvents?: OpenHandsEvent[];
  /**
   * `true` once an event outside this group has been emitted after it, so the
   * group is no longer the "live" tail of the chat. While `false` (the
   * default), the group keeps showing the most recent action's title as its
   * prominent summary, with the count of completed actions shown subtly on
   * the right.
   */
  isFinalized?: boolean;
  /** The fully-rendered event messages to show when the group is expanded. */
  children: React.ReactNode;
}

/**
 * Collapsible container that wraps a run of consecutive agent action/observation
 * events into a single summary card.
 *
 * Collapsed, while the group is still the live tail of the chat
 * (`isFinalized=false`):
 *   - Left (prominent): the title of the most recent action/observation in
 *     the group — i.e. either the action currently in flight, or the latest
 *     completed step.
 *   - Right (subdued):  "{completed}/{total} actions completed" while at
 *     least one action is still pending (with a spinner), otherwise
 *     "{count} actions completed" followed by a success check.
 *
 * Collapsed, after the group has been "moved past" (`isFinalized=true`):
 *   - "{count} actions completed" is promoted to the prominent foreground
 *     style and the count is the only thing shown next to the chevron.
 *
 * Expanded:
 *   - Renders the children verbatim, so each individual action/observation can
 *     still be expanded the way it was before grouping.
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

  // Each ObservationEvent in the group is a completed action. An ActionEvent
  // that's still here (i.e. not yet replaced by its observation in the UI
  // events array) is an action currently in flight.
  const pendingAction = events.find((e): e is ActionEvent => isActionEvent(e));
  const completedCount = events.filter(isObservationEvent).length;
  const totalCount = events.length;
  const isRunning = !!pendingAction;

  // Title of the most recent groupable event. While running this is the
  // pending action; otherwise it's the latest observation, with its
  // originating action looked up so the title can be the action-style summary
  // ("Editing path/to/file") instead of the observation default.
  const latestEvent = events[events.length - 1];
  let latestTitle: React.ReactNode = null;
  if (latestEvent) {
    if (isActionEvent(latestEvent)) {
      latestTitle = getEventContent(latestEvent).title;
    } else if (isObservationEvent(latestEvent)) {
      const lookupSource = allEvents ?? events;
      const correspondingAction = lookupSource.find(
        (e): e is ActionEvent =>
          isActionEvent(e) && e.id === latestEvent.action_id,
      );
      latestTitle = getEventContent(latestEvent, correspondingAction).title;
    }
  }

  const countSummary = isRunning
    ? t(I18nKey.EVENT_GROUP$ACTIONS_PROGRESS, {
        completed: completedCount,
        total: totalCount,
      })
    : t(I18nKey.EVENT_GROUP$ACTIONS_COMPLETED, { count: totalCount });

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
        className="w-full flex items-center justify-between gap-2 text-left cursor-pointer"
      >
        {isFinalized ? (
          <span className="flex items-center gap-2 min-w-0 font-normal text-[var(--oh-muted)]">
            <Chevron className="h-4 w-4 fill-[var(--oh-muted)] flex-shrink-0" />
            <span className="truncate">{countSummary}</span>
          </span>
        ) : (
          <>
            <span className="flex items-center gap-2 min-w-0 font-normal text-[var(--oh-muted)]">
              <Chevron className="h-4 w-4 fill-[var(--oh-muted)] flex-shrink-0" />
              <span className="truncate">{latestTitle ?? countSummary}</span>
            </span>
            <span className="flex items-center flex-shrink-0 font-normal text-[var(--oh-muted)]">
              <span className="truncate">{countSummary}</span>
              {isRunning ? (
                <LoaderCircle
                  data-testid="spinner-icon"
                  className="h-4 w-4 ml-2 inline animate-spin text-[var(--oh-muted)]"
                />
              ) : null}
            </span>
          </>
        )}
      </button>

      {expanded && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={buttonId}
          className="mt-1.5 flex flex-col"
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
