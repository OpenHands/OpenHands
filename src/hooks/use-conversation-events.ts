import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  EMPTY_EVENTS,
  useEventStore,
  type OHEvent,
} from "#/stores/use-event-store";

/**
 * Resolve which conversation a chat component is reading. An explicit id wins;
 * otherwise the id comes from the surrounding navigation scope. A popout
 * nests its own `NavigationProvider`, so the same components read that popout's
 * conversation without any of them taking a prop.
 */
export function useScopedConversationId(
  conversationId?: string | null,
): string | null {
  const { conversationId: contextConversationId } = useOptionalConversationId();
  return conversationId ?? contextConversationId;
}

/** The raw event stream for the conversation in scope. */
export function useConversationEvents(
  conversationId?: string | null,
): OHEvent[] {
  const id = useScopedConversationId(conversationId);
  return useEventStore((state) =>
    id ? (state.byConversation[id]?.events ?? EMPTY_EVENTS) : EMPTY_EVENTS,
  );
}

/** The UI-projected event stream for the conversation in scope. */
export function useConversationUiEvents(
  conversationId?: string | null,
): OHEvent[] {
  const id = useScopedConversationId(conversationId);
  return useEventStore((state) =>
    id ? (state.byConversation[id]?.uiEvents ?? EMPTY_EVENTS) : EMPTY_EVENTS,
  );
}

/** Whether the conversation in scope has been loaded into the store. */
export function useIsConversationLoaded(
  conversationId?: string | null,
): boolean {
  const id = useScopedConversationId(conversationId);
  return useEventStore((state) => (id ? !!state.byConversation[id] : false));
}
