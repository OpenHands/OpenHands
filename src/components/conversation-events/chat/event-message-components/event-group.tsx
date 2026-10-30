import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isAgentErrorEvent,
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
   * the summary counts it as the right kind of step. Falls back to `events`
   * when omitted.
   */
  allEvents?: OpenHandsEvent[];
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
 * Collapsed otherwise: what the steps did, e.g. "Ran 3 commands · Read 1 file".
 *
 * Expanded: the children in a bordered list, so each individual
 * action/observation can still be expanded the way it was before grouping.
 */
export function EventGroup({
  events,
  allEvents,
  isLive = false,
  open,
  onOpenChange,
  children,
}: EventGroupProps) {
  const { t } = useTranslation("openhands");
  const [localExpanded, setLocalExpanded] = React.useState(false);
  const expanded = open ?? localExpanded;
  const toggleExpanded = () =>
    onOpenChange ? onOpenChange(!expanded) : setLocalExpanded(!expanded);
  const contentId = React.useId();
  const buttonId = `${contentId}-toggle`;

  if (events.length === 0) {
    return null;
  }

  const lookupSource = allEvents ?? events;
  let summary: React.ReactNode = getEventGroupSummaryParts(events, lookupSource)
    .map(({ key, count }) => t(key, { count }))
    .join(" · ");

  if (isLive) {
    // An action still in the UI events is in flight, unless an AgentErrorEvent
    // resolved it (that event renders on its own and never replaces it).
    const erroredToolCallIds = new Set(
      lookupSource.filter(isAgentErrorEvent).map((e) => e.tool_call_id),
    );
    const running = events.find(
      (e): e is ActionEvent =>
        isActionEvent(e) && !erroredToolCallIds.has(e.tool_call_id),
    );
    summary = running ? (
      <PathInteractiveContext.Provider value={false}>
        {getEventContent(running).title}
      </PathInteractiveContext.Provider>
    ) : (
      t(I18nKey.ACTION_MESSAGE$THINK)
    );
  }

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
        {isLive && (
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
