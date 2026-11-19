import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";

/**
 * The latest edit of every edited message in one events array: target event
 * id -> the message's whole text after that edit. The agent edits its
 * earlier messages with its `edit_message` tool (the `message_edits`
 * experimental feature); the event log stays append-only, so each edit is a
 * `MessageEditEvent` naming its message, and the latest one wins.
 *
 * The store holds the newest events plus the older pages loaded so far, and an
 * edit always comes after its message, so every loaded message has its edits
 * loaded too: no extra request, and never the whole log. The index is built
 * once per events array, not once per message.
 */
const editIndexes = new WeakMap<object, Map<string, string>>();

function editIndex(events: readonly OHEvent[]): Map<string, string> {
  const cached = editIndexes.get(events);
  if (cached) return cached;
  const index = new Map<string, string>();
  for (const event of events) {
    if ("kind" in event && event.kind === "MessageEditEvent") {
      index.set(event.target_event_id, event.text);
    }
  }
  editIndexes.set(events, index);
  return index;
}

/** The text of the agent's message after its latest edit, if it was edited. */
export function useMessageEdit(
  eventId: string | undefined,
): string | undefined {
  return useEventStore((state) =>
    eventId ? editIndex(state.events).get(eventId) : undefined,
  );
}

/** The small "edited" mark under a message the agent edited. */
export function MessageEditedMark() {
  const { t } = useTranslation("openhands");
  return (
    <span
      data-testid="message-edited"
      className="mt-1 block text-xs italic text-text-tertiary"
    >
      {t(I18nKey.CHAT_INTERFACE$MESSAGE_EDITED)}
    </span>
  );
}
