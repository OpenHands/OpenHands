import { useMemo } from "react";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isUserMessageEvent,
  isPlanningFileEditorObservationEvent,
} from "#/types/agent-server/type-guards";

/**
 * Groups events into phases based on user messages.
 * A phase starts with a user message and includes all subsequent events
 * until the next user message.
 *
 * @param events - The full list of events
 * @returns Array of phases, where each phase is an array of events
 */
function groupEventsByPhase(events: OpenHandsEvent[]): OpenHandsEvent[][] {
  const phases: OpenHandsEvent[][] = [];
  let currentPhase: OpenHandsEvent[] = [];

  for (const event of events) {
    if (isUserMessageEvent(event)) {
      // Start a new phase with the user message
      if (currentPhase.length > 0) {
        phases.push(currentPhase);
      }
      currentPhase = [event];
    } else {
      // Add event to current phase
      currentPhase.push(event);
    }
  }

  // Don't forget the last phase
  if (currentPhase.length > 0) {
    phases.push(currentPhase);
  }

  return phases;
}

const isPlanFilePath = (path: string | null): boolean =>
  path?.toUpperCase().endsWith("PLAN.MD") ?? false;

/**
 * Finds the last PlanningFileEditorObservation for Plan.md in a phase.
 *
 * @param phase - Array of events in a phase
 * @returns The event ID of the last Plan.md observation, or null
 */
function findLastPlanningObservationInPhase(
  phase: OpenHandsEvent[],
): string | null {
  for (let i = phase.length - 1; i >= 0; i -= 1) {
    const event = phase[i];
    if (
      isPlanningFileEditorObservationEvent(event) &&
      isPlanFilePath(event.observation.path)
    ) {
      return event.id;
    }
  }
  return null;
}

export interface PlanPreviewEventInfo {
  eventId: string;
  /** Index of this plan preview in the conversation (1st, 2nd, etc.) */
  phaseIndex: number;
}

/**
 * Hook to determine which PlanningFileEditorObservation events should render PlanPreview.
 *
 * This hook implements phase-based grouping where:
 * - A phase starts with a user message and ends at the next user message
 * - Only the LAST PlanningFileEditorObservation in each phase shows PlanPreview
 * - This ensures only one preview per user request, even with multiple observations
 *
 * Scenario handling:
 * - Scenario 1 (Create plan): Multiple observations in one phase → 1 preview
 * - Scenario 2 (Create then update): Two user messages → two phases → 2 previews
 * - Scenario 3 (Create + update while processing): Two user messages → 2 previews
 *
 * @param allEvents - Full list of v1 events (for phase detection)
 * @returns Set of event IDs that should render PlanPreview
 */
export function usePlanPreviewEvents(allEvents: OpenHandsEvent[]): Set<string> {
  // Derive a stable string key from the collected plan-preview ids, keyed on
  // the *content* of the result (which ids were selected) rather than the
  // `allEvents` array reference. During streaming, every token produces a fresh
  // `allEvents` ref even though only the last event's content changed, so this
  // memo recomputes — but the returned string is value-equal across tokens as
  // long as no new Plan.md observation arrived.
  const planPreviewKey = useMemo(() => {
    const phases = groupEventsByPhase(allEvents);
    const ids: string[] = [];
    phases.forEach((phase) => {
      const lastPlanningObservationId =
        findLastPlanningObservationInPhase(phase);
      if (lastPlanningObservationId) ids.push(lastPlanningObservationId);
    });
    return ids.join("\n");
  }, [allEvents]);

  // Strings compare by value in useMemo deps, so when no new Plan.md
  // observation arrived `planPreviewKey` is value-equal and the Set keeps its
  // identity across streaming tokens.
  return useMemo(() => {
    const planPreviewEventIds = new Set<string>();
    if (planPreviewKey) {
      planPreviewKey.split("\n").forEach((id) => planPreviewEventIds.add(id));
    }
    return planPreviewEventIds;
  }, [planPreviewKey]);
}

/**
 * Check if a specific event should render PlanPreview.
 *
 * @param eventId - The event ID to check
 * @param planPreviewEventIds - Set of event IDs that should render PlanPreview
 * @returns true if this event should render PlanPreview
 */
export function shouldShowPlanPreview(
  eventId: string,
  planPreviewEventIds: Set<string>,
): boolean {
  return planPreviewEventIds.has(eventId);
}
