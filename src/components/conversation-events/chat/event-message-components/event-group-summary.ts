import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isACPToolCallEvent,
  isHookExecutionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { buildActionById } from "../event-thought-helpers";

type StepKind = "command" | "read" | "edit" | "search" | "hook" | "tool";

// Base keys of `_one` / `_other` plural entries; i18next picks the form.
const STEP_KIND_KEYS: Record<StepKind, string> = {
  command: "EVENT_GROUP$RAN_COMMANDS",
  read: "EVENT_GROUP$READ_FILES",
  edit: "EVENT_GROUP$MADE_EDITS",
  search: "EVENT_GROUP$RAN_SEARCHES",
  hook: "EVENT_GROUP$RAN_HOOKS",
  tool: "EVENT_GROUP$USED_TOOLS",
};

const kindFromName = (kind: string, command?: unknown): StepKind => {
  switch (kind) {
    case "ExecuteBashAction":
    case "TerminalAction":
    case "ExecuteBashObservation":
    case "TerminalObservation":
      return "command";
    case "FileEditorAction":
    case "StrReplaceEditorAction":
    case "FileEditorObservation":
    case "StrReplaceEditorObservation":
      return command === "view" ? "read" : "edit";
    case "GrepAction":
    case "GlobAction":
    case "GrepObservation":
    case "GlobObservation":
      return "search";
    default:
      return "tool";
  }
};

const getStepKind = (
  event: OpenHandsEvent,
  actionById: ReadonlyMap<string, ActionEvent>,
): StepKind => {
  if (isHookExecutionEvent(event)) return "hook";
  if (isActionEvent(event)) {
    return kindFromName(
      event.action.kind,
      "command" in event.action ? event.action.command : undefined,
    );
  }
  if (isObservationEvent(event) || isUserRejectObservation(event)) {
    const action = actionById.get(event.action_id ?? "");
    if (action) return getStepKind(action, actionById);
    if (isObservationEvent(event)) {
      const { observation } = event;
      return kindFromName(
        observation.kind,
        "command" in observation ? observation.command : undefined,
      );
    }
  }
  return "tool";
};

/**
 * Counts a group's steps by what they did ("Ran 3 commands", "Read 2 files"),
 * in order of first appearance, as i18n key/count pairs.
 */
export const getEventGroupSummaryParts = (
  events: OpenHandsEvent[],
  lookupSource: OpenHandsEvent[] = events,
  actionById: ReadonlyMap<string, ActionEvent> = buildActionById(lookupSource),
): { key: string; count: number }[] => {
  const counts = new Map<StepKind, number>();
  events.forEach((event) => {
    const action = isObservationEvent(event)
      ? actionById.get(event.action_id ?? "")
      : event;
    if (action && isActionEvent(action) && action.action.kind === "ThinkAction")
      return;
    const kind = getStepKind(event, actionById);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  });
  return Array.from(counts, ([kind, count]) => ({
    key: STEP_KIND_KEYS[kind],
    count,
  }));
};

export const getSingleToolEvent = (events: OpenHandsEvent[]) => {
  const steps = events.filter(
    (event) =>
      !(
        (isActionEvent(event) && event.action.kind === "ThinkAction") ||
        (isObservationEvent(event) &&
          event.observation.kind === "ThinkObservation")
      ),
  );
  if (steps.length !== 1) return undefined;
  const event = steps[0];
  return isActionEvent(event) ||
    isObservationEvent(event) ||
    isUserRejectObservation(event) ||
    isACPToolCallEvent(event)
    ? event
    : undefined;
};
