import type { OpenHandsEvent } from "#/types/agent-server/core";
import type { SlashCommandOutput } from "#/stores/slash-command-output-store";
import { isPendingUserMessageBoundary } from "#/hooks/chat/slash-command-timeline-boundary";

export interface SlashCommandOutputPlacements {
  entriesBeforeEvent: Map<string, SlashCommandOutput[]>;
  tailEntries: SlashCommandOutput[];
  unresolvedActiveEntries: SlashCommandOutput[];
  breakBeforeEventIds: ReadonlySet<string>;
}

/**
 * Resolve raw-timeline command boundaries into positions in the current UI
 * event list. A boundary's original UI event may have been replaced, so output
 * is placed before the first currently rendered event that follows it in the
 * raw timeline, or at the rendered tail when its loaded boundary has no later
 * event. A boundary outside the current history window remains unresolved and
 * hidden until pagination restores it.
 *
 * Null-boundary entries belong to the existing empty-conversation/home slot
 * and are intentionally left to those renderers.
 */
export const resolveSlashCommandOutputPlacements = (
  entries: SlashCommandOutput[],
  allEvents: OpenHandsEvent[],
  renderedEvents: OpenHandsEvent[],
): SlashCommandOutputPlacements => {
  const rawIndexById = new Map<string, number>();
  allEvents.forEach((event, index) =>
    rawIndexById.set(String(event.id), index),
  );

  const renderedTimelineEvents = renderedEvents.flatMap((event) => {
    const rawIndex = rawIndexById.get(String(event.id));
    return rawIndex === undefined
      ? []
      : [{ eventId: String(event.id), rawIndex }];
  });

  const entriesBeforeEvent = new Map<string, SlashCommandOutput[]>();
  const tailEntries: SlashCommandOutput[] = [];
  const unresolvedActiveEntries: SlashCommandOutput[] = [];

  for (const entry of entries) {
    if (entry.timelineBoundaryEventId === null) continue;
    // Optimistic-message boundaries are rendered beside their pending bubble,
    // then atomically re-anchored to the server echo. Treating them as an
    // ordinary unresolved history ID would place `/skills` before the prompt.
    if (isPendingUserMessageBoundary(entry.timelineBoundaryEventId)) continue;

    const boundaryIndex = rawIndexById.get(entry.timelineBoundaryEventId);

    // A non-null boundary can fall outside the currently loaded history
    // window after a conversation switch. Its position is unknown until an
    // older page restores that raw event, so rendering it at the current tail
    // would move historical command output after newer conversation content.
    if (boundaryIndex === undefined) {
      if (entry.kind === "skills" && entry.showWhenPlacementUnresolved) {
        unresolvedActiveEntries.push(entry);
      }
      continue;
    }

    const firstLaterEvent = renderedTimelineEvents.find(
      ({ rawIndex }) => rawIndex > boundaryIndex,
    );

    if (!firstLaterEvent) {
      tailEntries.push(entry);
      continue;
    }

    const existing = entriesBeforeEvent.get(firstLaterEvent.eventId) ?? [];
    existing.push(entry);
    entriesBeforeEvent.set(firstLaterEvent.eventId, existing);
  }

  return {
    entriesBeforeEvent,
    tailEntries,
    unresolvedActiveEntries,
    breakBeforeEventIds: new Set(entriesBeforeEvent.keys()),
  };
};
