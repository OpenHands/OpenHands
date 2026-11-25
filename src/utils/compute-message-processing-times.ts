import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isMessageEvent,
  isUserMessageEvent,
} from "#/types/agent-server/type-guards";
import { parseDateAsUTC } from "#/utils/format-time-delta";

const toTimestampMs = (timestamp?: string): number | null => {
  if (!timestamp) return null;
  const ms = parseDateAsUTC(timestamp).getTime();
  return Number.isFinite(ms) ? ms : null;
};

/**
 * Maps each assistant message event id to how long its request took: the
 * elapsed time (ms) between the most recent preceding user message and that
 * reply. Only `source === "agent"` message events receive an entry; a reply
 * with no preceding user message (e.g. history is only partially loaded) is
 * skipped.
 */
export const computeAssistantProcessingTimes = (
  events: OpenHandsEvent[],
): Map<string, number> => {
  const processingTimes = new Map<string, number>();
  let lastUserTimestampMs: number | null = null;

  for (const event of events) {
    if (isUserMessageEvent(event)) {
      lastUserTimestampMs = toTimestampMs(event.timestamp);
      continue;
    }

    if (
      isMessageEvent(event) &&
      event.source === "agent" &&
      lastUserTimestampMs !== null
    ) {
      const replyMs = toTimestampMs(event.timestamp);
      if (replyMs !== null) {
        processingTimes.set(
          event.id,
          Math.max(0, replyMs - lastUserTimestampMs),
        );
      }
    }
  }

  return processingTimes;
};
