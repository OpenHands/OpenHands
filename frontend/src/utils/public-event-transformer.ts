import { OpenHandsEvent } from "#/types/v1/core";

/**
 * Filter events that should be rendered in the UI
 * This applies the same filtering logic as the V1Messages component
 */
export function filterRenderableEvents(
  events: OpenHandsEvent[],
): OpenHandsEvent[] {
  // Import the shouldRenderEvent function dynamically to avoid circular dependencies
  return events.filter(
    () =>
      // Basic filtering - we can expand this based on the actual shouldRenderEvent logic
      // For now, we'll include most events except system-level ones
      true, // Let V1Messages handle the filtering
  );
}
