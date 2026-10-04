import { create } from "zustand";
import { useEventStore } from "#/stores/use-event-store";
import { parseDateAsUTC } from "#/utils/format-time-delta";
import { matchesPendingConversationId } from "#/utils/pending-task-message-link";
import type { BaseEvent } from "#/types/agent-server/core/base/event";

export type PendingUserMessageStatus = "sending" | "error";

/**
 * How long a pending message is allowed to stay in "sending" state before we
 * give up and flip it to "error" with a retry link. This guards against the
 * "server crashed / websocket dropped after our send resolved, echo never
 * arrives" scenario where the message would otherwise hang forever.
 *
 * Exported so tests can override it via vi.fakeTimers without hard-coding the
 * value.
 */
export const PENDING_MESSAGE_TIMEOUT_MS = 150_000;

export interface PendingUserMessage {
  id: string;
  /**
   * The conversation this pending message belongs to. The chat UI filters the
   * global queue by the active conversation id so messages enqueued in one
   * conversation never leak into another when the user switches.
   */
  conversationId: string;
  /** User-visible bubble text (what the user typed; no file annotations). */
  text: string;
  /**
   * The exact string sent to the server (may include the appended
   * "Files uploaded: …" prompt when attachments are present). Used as the
   * primary key when matching against the echoed `UserMessageEvent`.
   */
  content: string;
  status: PendingUserMessageStatus;
  imageUrls: string[];
  fileUrls: string[];
  timestamp: string;
  afterTimestamp?: number;
  errorMessage?: string;
}

interface OptimisticUserMessageState {
  pendingMessages: PendingUserMessage[];
  confirmedEventIds: Set<string>;
}

export interface EnqueuePendingMessagePayload {
  conversationId: string;
  /** User-visible text for the bubble. */
  text: string;
  /**
   * The exact string sent to the server. Defaults to `text` for call sites
   * that don't transform the content (e.g. git-control-bar, task-card).
   */
  content?: string;
  imageUrls?: string[];
  fileUrls?: string[];
  timestamp?: string;
}

interface OptimisticUserMessageActions {
  /**
   * Append a new user message to the queue with status "sending".
   * Returns the locally-generated id for later updates. Schedules a
   * `PENDING_MESSAGE_TIMEOUT_MS` watchdog that flips the entry to "error" if
   * it's still in "sending" state when the timer fires.
   */
  enqueuePendingMessage: (payload: EnqueuePendingMessagePayload) => string;
  /** Mark a pending message as failed (the API rejected it). */
  markPendingMessageError: (id: string, errorMessage?: string) => void;
  /** Mark a pending message as sending again (used when retrying). */
  markPendingMessageSending: (id: string) => void;
  /** Drop a pending message from the queue (e.g., after success/cancellation). */
  removePendingMessage: (id: string) => void;
  /** Match conversation (including task links), server history and content once per event.
   * Prefer a sending attempt; an errored attempt may still receive a late echo. */
  consumeMatchingPendingMessage: (
    conversationId: string,
    content: string,
    event: Pick<BaseEvent, "id" | "timestamp">,
  ) => PendingUserMessage | null;
  /** Wipe all queued messages (e.g., when changing conversations). */
  clearPendingMessages: () => void;
  /**
   * Move pending entries from a provisional task URL (`task-{uuid}`) to the
   * real conversation id once cloud provisioning finishes.
   */
  reassignPendingMessages: (
    fromConversationId: string,
    toConversationId: string,
  ) => void;
}

type OptimisticUserMessageStore = OptimisticUserMessageState &
  OptimisticUserMessageActions;

const initialState: OptimisticUserMessageState = {
  pendingMessages: [],
  confirmedEventIds: new Set(),
};

// Use a timestamp + random suffix instead of a module-level counter so ids
// stay unique across test resets and don't accumulate state between runs.
// `crypto.randomUUID` would be ideal but isn't available in older test
// environments, so a base36 random suffix is a safe lowest-common-denominator.
const generatePendingId = (): string =>
  `pending-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

const latestServerTimestamp = (conversationId: string): number | undefined => {
  const { events, loadedConversationId } = useEventStore.getState();
  if (
    !loadedConversationId ||
    !matchesPendingConversationId(loadedConversationId, conversationId)
  )
    return undefined;

  let latest: number | undefined;
  for (const event of events) {
    if (!("timestamp" in event) || !event.timestamp) continue;
    const timestamp = parseDateAsUTC(event.timestamp).getTime();
    if (
      Number.isFinite(timestamp) &&
      (latest === undefined || timestamp > latest)
    ) {
      latest = timestamp;
    }
  }
  return latest;
};

export const useOptimisticUserMessageStore = create<OptimisticUserMessageStore>(
  (set, get) => ({
    ...initialState,

    enqueuePendingMessage: (payload) => {
      const id = generatePendingId();
      const message: PendingUserMessage = {
        id,
        conversationId: payload.conversationId,
        text: payload.text,
        content: payload.content ?? payload.text,
        status: "sending",
        imageUrls: payload.imageUrls ?? [],
        fileUrls: payload.fileUrls ?? [],
        timestamp: payload.timestamp ?? new Date().toISOString(),
        afterTimestamp: latestServerTimestamp(payload.conversationId),
      };
      set((state) => ({
        pendingMessages: [...state.pendingMessages, message],
      }));

      // Watchdog: if the server echo never lands (WS dropped, server crashed,
      // network partition), flip this entry to "error" so the user gets a
      // retry link instead of a permanently-pinned "Sending…" bubble.
      setTimeout(() => {
        const current = get().pendingMessages.find((m) => m.id === id);
        if (current?.status === "sending") {
          get().markPendingMessageError(id, "Send timed out");
        }
      }, PENDING_MESSAGE_TIMEOUT_MS);

      return id;
    },

    markPendingMessageError: (id, errorMessage) =>
      set((state) => ({
        pendingMessages: state.pendingMessages.map((message) =>
          message.id === id
            ? { ...message, status: "error", errorMessage }
            : message,
        ),
      })),

    markPendingMessageSending: (id) =>
      set((state) => ({
        pendingMessages: state.pendingMessages.map((message) =>
          message.id === id
            ? {
                ...message,
                status: "sending",
                errorMessage: undefined,
                afterTimestamp: latestServerTimestamp(message.conversationId),
              }
            : message,
        ),
      })),

    removePendingMessage: (id) =>
      set((state) => ({
        pendingMessages: state.pendingMessages.filter(
          (message) => message.id !== id,
        ),
      })),

    consumeMatchingPendingMessage: (conversationId, content, event) => {
      let consumed: PendingUserMessage | null = null;
      const confirmationId = `${conversationId}:${event.id}`;
      const confirmedAt = parseDateAsUTC(event.timestamp).getTime();
      set((state) => {
        if (state.confirmedEventIds.has(confirmationId)) return state;
        // Browser clocks cannot establish ordering against server event history.
        const matches = (message: PendingUserMessage) =>
          matchesPendingConversationId(
            conversationId,
            message.conversationId,
          ) &&
          (message.afterTimestamp === undefined ||
            confirmedAt > message.afterTimestamp) &&
          message.content.trim() === content.trim();
        let target = state.pendingMessages.findIndex(
          (message) => message.status === "sending" && matches(message),
        );
        if (target === -1) target = state.pendingMessages.findIndex(matches);
        if (target === -1) return state;
        consumed = state.pendingMessages[target];
        return {
          pendingMessages: state.pendingMessages.filter(
            (_, index) => index !== target,
          ),
          confirmedEventIds: new Set([
            ...state.confirmedEventIds,
            confirmationId,
          ]),
        };
      });
      return consumed;
    },

    clearPendingMessages: () =>
      set(() => ({ pendingMessages: [], confirmedEventIds: new Set() })),

    reassignPendingMessages: (fromConversationId, toConversationId) =>
      set((state) => ({
        pendingMessages: state.pendingMessages.map((message) =>
          message.conversationId === fromConversationId
            ? { ...message, conversationId: toConversationId }
            : message,
        ),
      })),
  }),
);
