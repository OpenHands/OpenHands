import { useEffect, useRef } from "react";

import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

const FILE_MUTATION_KINDS = new Set([
  "FileEditorObservation",
  "StrReplaceEditorObservation",
  "PlanningFileEditorObservation",
  "ExecuteBashObservation",
  "TerminalObservation",
]);

function isWorkspaceMutation(event: OHEvent): boolean {
  const observation = (
    event as { observation?: { kind?: string; command?: string } }
  ).observation;
  if (!observation || !FILE_MUTATION_KINDS.has(observation.kind ?? "")) {
    return false;
  }
  return !(
    [
      "FileEditorObservation",
      "StrReplaceEditorObservation",
      "PlanningFileEditorObservation",
    ].includes(observation.kind ?? "") && observation.command === "view"
  );
}

/**
 * Bumps the shared workspace version when the agent has actually changed disk.
 * The preview consumes the version in its iframe URL, so the browser performs
 * a real re-fetch instead of relying on a synthetic preview state.
 */
export function useAutoRefreshPreviewOnEdit(): void {
  const events = useEventStore((state) => state.events);
  const bumpWorkspaceMutationCounter = useWorkspaceMutationCounter(
    (state) => state.bump,
  );
  const processedIdsRef = useRef<Set<string | number>>(new Set());
  const processedEventsRef = useRef<WeakSet<OHEvent>>(new WeakSet());

  useEffect(() => {
    let hasNewMutation = false;

    for (const event of events) {
      const id: string | number | undefined =
        "id" in event ? event.id : undefined;
      const alreadyProcessed =
        id !== undefined
          ? processedIdsRef.current.has(id)
          : processedEventsRef.current.has(event);

      if (!alreadyProcessed) {
        if (id !== undefined) {
          processedIdsRef.current.add(id);
        } else {
          processedEventsRef.current.add(event);
        }
        if (isWorkspaceMutation(event)) hasNewMutation = true;
      }
    }

    if (hasNewMutation) bumpWorkspaceMutationCounter();
  }, [events, bumpWorkspaceMutationCounter]);
}
