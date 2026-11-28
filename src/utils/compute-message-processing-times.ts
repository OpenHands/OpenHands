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
 * Maps the LAST assistant message of each turn to how long that request took:
 * the elapsed time (ms) between the turn's user message and that final reply.
 * Intermediate assistant messages in a turn (progress updates) receive no
 * entry, so a single "took ~Xs" appears per request rather than one under every
 * update. A reply with no preceding user message (e.g. history only partially
 * loaded) is skipped.
 */
export const computeAssistantProcessingTimes = (
  events: OpenHandsEvent[],
): Map<string, number> => {
  const processingTimes = new Map<string, number>();
  let turnStartMs: number | null = null;
  let lastReplyId: string | null = null;
  let lastReplyMs: number | null = null;

  const finalizeTurn = () => {
    if (turnStartMs !== null && lastReplyId !== null && lastReplyMs !== null) {
      processingTimes.set(lastReplyId, Math.max(0, lastReplyMs - turnStartMs));
    }
  };

  for (const event of events) {
    if (isUserMessageEvent(event)) {
      finalizeTurn();
      turnStartMs = toTimestampMs(event.timestamp);
      lastReplyId = null;
      lastReplyMs = null;
      continue;
    }

    if (
      isMessageEvent(event) &&
      event.source === "agent" &&
      turnStartMs !== null
    ) {
      const replyMs = toTimestampMs(event.timestamp);
      if (replyMs !== null) {
        lastReplyId = event.id;
        lastReplyMs = replyMs;
      }
    }
  }

  finalizeTurn();
  return processingTimes;
};
