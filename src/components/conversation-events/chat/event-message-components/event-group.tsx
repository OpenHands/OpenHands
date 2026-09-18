import React from "react";
import { useTranslation } from "react-i18next";
import { BookOpen, LoaderCircle, Pencil, Terminal, Wrench } from "lucide-react";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { OpenHandsEvent, ActionEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import { I18nKey } from "#/i18n/declaration";
import { IsInEventGroupContext } from "../../../features/chat/is-in-event-group-context";

type ActionCategory = "read" | "edit" | "command" | "other";

const getActionCategory = (event: ActionEvent): ActionCategory => {
  const { action } = event;

  if (
    action.kind === "FileEditorAction" ||
    action.kind === "StrReplaceEditorAction"
  ) {
    return action.command === "view" ? "read" : "edit";
  }

  if (
    action.kind === "GlobAction" ||
    action.kind === "GrepAction" ||
    action.kind.startsWith("Browser")
  ) {
    return "read";
  }

  if (action.kind === "ExecuteBashAction" || action.kind === "TerminalAction") {
    return "command";
  }

  return "other";
};

const getObservationCategory = (event: OpenHandsEvent): ActionCategory => {
  if (!isObservationEvent(event)) return "other";

  if (
    event.observation.kind === "FileEditorObservation" ||
    event.observation.kind === "StrReplaceEditorObservation"
  ) {
    return "edit";
  }
  if (
    event.observation.kind === "ExecuteBashObservation" ||
    event.observation.kind === "TerminalObservation"
  ) {
    return "command";
  }
  if (event.observation.kind.startsWith("Browser")) return "read";

  return "other";
};

const CATEGORY_CONFIG = {
  read: { icon: BookOpen, key: "EVENT_GROUP$READ_COUNT" },
  edit: { icon: Pencil, key: "EVENT_GROUP$EDIT_COUNT" },
  command: { icon: Terminal, key: "EVENT_GROUP$COMMAND_COUNT" },
  other: { icon: Wrench, key: "EVENT_GROUP$OTHER_COUNT" },
} as const;

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
  /** Retained for callers that distinguish live and finalized groups. */
  isFinalized?: boolean;
  /** The fully-rendered event messages to show when the group is expanded. */
  children: React.ReactNode;
}

/**
 * Collapsible container that wraps a run of consecutive agent action/observation
 * events into a single summary card.
 *
 * Collapsed, it summarizes every represented event by user-facing action
 * categories with distinct icons for reads, edits, commands, and other
 * actions. A live group keeps its running spinner while an action is pending.
 *
 * Expanded:
 *   - Renders the children verbatim, so each individual action/observation can
 *     still be expanded the way it was before grouping.
 */
export function EventGroup({ events, allEvents, children }: EventGroupProps) {
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

  const lookupSource = allEvents ?? events;
  const categories: Record<ActionCategory, number> = {
    read: 0,
    edit: 0,
    command: 0,
    other: 0,
  };
  const countedActionIds = new Set<string>();
  events.forEach((event, index) => {
    const action = isActionEvent(event)
      ? event
      : isObservationEvent(event)
        ? lookupSource.find(
            (candidate): candidate is ActionEvent =>
              isActionEvent(candidate) && candidate.id === event.action_id,
          )
        : undefined;
    const actionId =
      action?.id ??
      (isObservationEvent(event) ? event.action_id : undefined) ??
      event.id ??
      `event-${index}`;
    if (countedActionIds.has(actionId)) return;

    countedActionIds.add(actionId);
    const category = action
      ? getActionCategory(action)
      : getObservationCategory(event);
    categories[category] += 1;
  });

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
        <span className="flex items-center gap-2 min-w-0 font-normal text-[var(--oh-muted)]">
          <Chevron className="h-4 w-4 fill-[var(--oh-muted)] flex-shrink-0" />
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {(Object.keys(CATEGORY_CONFIG) as ActionCategory[]).map(
              (category) => {
                const count = categories[category];
                if (count === 0) return null;
                const { icon: Icon, key } = CATEGORY_CONFIG[category];
                return (
                  <span
                    key={category}
                    className="inline-flex items-center gap-1 whitespace-nowrap"
                    data-testid={`event-group-${category}-count`}
                  >
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                    {t(`${key}_${count === 1 ? "one" : "other"}`, { count })}
                  </span>
                );
              },
            )}
          </span>
        </span>
        {isRunning ? (
          <LoaderCircle
            data-testid="spinner-icon"
            aria-label={countSummary}
            className="h-4 w-4 inline flex-shrink-0 animate-spin text-[var(--oh-muted)]"
          />
        ) : null}
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
