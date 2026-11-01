import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isHookExecutionEvent,
  isObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";

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
  lookupSource: OpenHandsEvent[],
): StepKind => {
  if (isHookExecutionEvent(event)) return "hook";
  if (isActionEvent(event)) {
    return kindFromName(
      event.action.kind,
      "command" in event.action ? event.action.command : undefined,
    );
  }
  if (isObservationEvent(event) || isUserRejectObservation(event)) {
    const action = lookupSource.find(
      (e): e is ActionEvent => isActionEvent(e) && e.id === event.action_id,
    );
    if (action) return getStepKind(action, lookupSource);
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
): { key: string; count: number }[] => {
  const counts = new Map<StepKind, number>();
  events.forEach((event) => {
    const kind = getStepKind(event, lookupSource);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  });
  return Array.from(counts, ([kind, count]) => ({
    key: STEP_KIND_KEYS[kind],
    count,
  }));
};
