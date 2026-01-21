import { OpenHandsEvent } from "#/types/v1/core";
import { isObservationEvent, isActionEvent } from "#/types/v1/type-guards";

/**
 * Handles adding an event to the UI events array
 * Replaces actions with observations when they arrive (so UI shows observation instead of action)
 * Exception: ThinkAction is NOT replaced because the thought content is in the action, not in the observation
 */
export const handleEventForUI = (
  event: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] => {
  const newUiEvents = [...uiEvents];

  if (isObservationEvent(event)) {
    // Find and replace the corresponding action from uiEvents
    const actionIndex = newUiEvents.findIndex(
      (uiEvent) => uiEvent.id === event.action_id,
    );
    if (actionIndex !== -1) {
      const correspondingAction = newUiEvents[actionIndex];
      // Don't replace ThinkAction with ThinkObservation
      // The thought content is in the action, not the observation
      if (
        isActionEvent(correspondingAction) &&
        correspondingAction.action.kind === "ThinkAction"
      ) {
        return newUiEvents;
      }
      newUiEvents[actionIndex] = event;
    } else {
      // Action not found in uiEvents, just add the observation
      newUiEvents.push(event);
    }
  } else {
    // For non-observation events, just add them to uiEvents
    newUiEvents.push(event);
  }

  return newUiEvents;
};
