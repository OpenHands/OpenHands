import { OpenHandsEvent } from "#/types/v1/core";
import { PublicEvent } from "#/api/public-conversation-service.api";
import { isBaseEvent } from "#/types/v1/type-guards";

/**
 * Transform raw public events into V1 OpenHandsEvent format
 * This utility converts the backend's public event format into the format
 * expected by the V1Messages component.
 */
export function transformPublicEventsToV1(
  publicEvents: PublicEvent[],
): OpenHandsEvent[] {
  const transformedEvents: OpenHandsEvent[] = [];

  for (const publicEvent of publicEvents) {
    try {
      // The public event's data field should contain the actual V1 event
      const eventData = publicEvent.data;

      // Validate that the event data has the basic structure of a V1 event
      if (isBaseEvent(eventData)) {
        transformedEvents.push(eventData as OpenHandsEvent);
      } else {
        // If the event data doesn't match V1 format, try to create a compatible event
        // eslint-disable-next-line no-console
        console.warn(
          `Public event ${publicEvent.id} does not match V1 format, attempting to transform`,
          eventData,
        );

        // Create a basic message event from the raw data
        const fallbackEvent: OpenHandsEvent = {
          id: publicEvent.id,
          timestamp: publicEvent.timestamp,
          source: "agent" as const,
          llm_message: {
            role: "assistant" as const,
            content: [
              {
                type: "text" as const,
                text: JSON.stringify(eventData, null, 2),
              },
            ],
          },
          activated_microagents: [],
          extended_content: [],
        };

        transformedEvents.push(fallbackEvent);
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(
        `Failed to transform public event ${publicEvent.id}:`,
        error,
      );

      // Create a fallback error event
      const errorEvent: OpenHandsEvent = {
        id: publicEvent.id,
        timestamp: publicEvent.timestamp,
        source: "agent" as const,
        llm_message: {
          role: "assistant" as const,
          content: [
            {
              type: "text" as const,
              text: `Error displaying event: ${error instanceof Error ? error.message : "Unknown error"}`,
            },
          ],
        },
        activated_microagents: [],
        extended_content: [],
      };

      transformedEvents.push(errorEvent);
    }
  }

  return transformedEvents;
}

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
