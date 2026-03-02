import { useMemo } from "react";
import { OpenHandsEvent } from "#/types/v1/core";
import {
  isUserMessageEvent,
  isPlanningFileEditorObservationEvent,
} from "#/types/v1/type-guards";

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

/**
 * Finds the last PlanningFileEditorObservation in a phase.
 *
 * @param phase - Array of events in a phase
 * @returns The event ID of the last PlanningFileEditorObservation, or null
 */
function findLastPlanningObservationInPhase(
  phase: OpenHandsEvent[],
): string | null {
  // Iterate backwards to find the last one
  for (let i = phase.length - 1; i >= 0; i -= 1) {
    const event = phase[i];
    if (isPlanningFileEditorObservationEvent(event)) {
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
  return useMemo(() => {
    const planPreviewEventIds = new Set<string>();

    // Group events by phases (user message boundaries)
    const phases = groupEventsByPhase(allEvents);

    // For each phase, find the last PlanningFileEditorObservation
    phases.forEach((phase) => {
      const lastPlanningObservationId =
        findLastPlanningObservationInPhase(phase);
      if (lastPlanningObservationId) {
        planPreviewEventIds.add(lastPlanningObservationId);
      }
    });

    return planPreviewEventIds;
  }, [allEvents]);
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
