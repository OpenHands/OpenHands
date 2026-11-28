import { OpenHandsEvent } from "#/types/agent-server/core";
import { isMessageEvent } from "#/types/agent-server/type-guards";
import { parseDateAsUTC } from "#/utils/format-time-delta";

const TIMESTAMP_GAP_THRESHOLD_MS = 5 * 60_000;

const toDate = (timestamp?: string): Date | null => {
  if (!timestamp) return null;
  const date = parseDateAsUTC(timestamp);
  return Number.isNaN(date.getTime()) ? null : date;
};

export const isNewLocalDay = (previous: Date, current: Date): boolean =>
  previous.getFullYear() !== current.getFullYear() ||
  previous.getMonth() !== current.getMonth() ||
  previous.getDate() !== current.getDate();

/**
 * Returns the ids of chat messages whose timestamp should be shown by default:
 * those separated from the previous chat message by a real gap (5+ minutes) or
 * a new local day. Other messages keep hover-only timestamps.
 */
export const computeMessageGaps = (events: OpenHandsEvent[]): Set<string> => {
  const gapMessageIds = new Set<string>();
  let previousDate: Date | null = null;

  for (const event of events) {
    if (
      !isMessageEvent(event) ||
      (event.source !== "user" && event.source !== "agent")
    ) {
      continue;
    }

    const currentDate = toDate(event.timestamp);
    if (!currentDate) continue;

    if (
      previousDate !== null &&
      (currentDate.getTime() - previousDate.getTime() >
        TIMESTAMP_GAP_THRESHOLD_MS ||
        isNewLocalDay(previousDate, currentDate))
    ) {
      gapMessageIds.add(event.id);
    }

    previousDate = currentDate;
  }

  return gapMessageIds;
};
