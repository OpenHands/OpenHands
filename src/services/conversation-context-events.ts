import type { CanvasConversationContextChangeRequest } from "#/types/canvas-extension";

interface ConversationContextScope {
  backendId: string;
  orgId: string | null;
  connectionRevision: number;
}

type Listener = (event: CanvasConversationContextChangeRequest) => void;
const subscriptions = new Set<{
  scope: ConversationContextScope;
  listener: Listener;
}>();

/** Transient UI actions only; this is neither persisted state nor an API ack. */
export function notifyConversationContextChangeRequested(
  scope: ConversationContextScope,
  event: CanvasConversationContextChangeRequest,
) {
  for (const subscription of subscriptions) {
    if (
      subscription.scope.backendId === scope.backendId &&
      subscription.scope.orgId === scope.orgId &&
      subscription.scope.connectionRevision === scope.connectionRevision
    ) {
      try {
        subscription.listener(event);
      } catch {
        // An App must not prevent the user's context action or other listeners.
      }
    }
  }
}

export function subscribeConversationContextChangeRequested(
  scope: ConversationContextScope,
  listener: Listener,
) {
  const subscription = { scope: { ...scope }, listener };
  subscriptions.add(subscription);
  return () => {
    subscriptions.delete(subscription);
  };
}
