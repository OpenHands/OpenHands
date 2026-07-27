import { create } from "zustand";

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
  errorMessage?: string;
}

interface OptimisticUserMessageState {
  pendingMessages: PendingUserMessage[];
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
  /**
   * Remove the pending message that matches the given echoed `content` in
   * the given conversation. Matching is done by exact content equality on
   * messages still in "sending" state; if no match exists we fall back to
   * removing the oldest "sending" entry in that conversation so that an echo
   * with a slightly munged body (e.g. trailing-whitespace stripped by the
   * server) still clears its bubble. Scoping by `conversationId` ensures a
   * stale ack for one conversation never pops a pending entry belonging to
   * another.
   */
  consumeMatchingPendingMessage: (
    conversationId: string,
    content: string,
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
};

// Use a timestamp + random suffix instead of a module-level counter so ids
// stay unique across test resets and don't accumulate state between runs.
// `crypto.randomUUID` would be ideal but isn't available in older test
// environments, so a base36 random suffix is a safe lowest-common-denominator.
const generatePendingId = (): string =>
  `pending-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

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
            ? { ...message, status: "sending", errorMessage: undefined }
            : message,
        ),
      })),

    removePendingMessage: (id) =>
      set((state) => ({
        pendingMessages: state.pendingMessages.filter(
          (message) => message.id !== id,
        ),
      })),

    consumeMatchingPendingMessage: (conversationId, content) => {
      // Single atomic `set` so the find + filter can't observe an interleaved
      // mutation from another action. We prefer an exact content match (this
      // is what makes out-of-order echoes safe: an echo of "world" will pop
      // the "world" bubble, not the older "hello" one). If no exact match
      // exists — e.g. the server slightly munged the body — fall back to the
      // oldest "sending" entry in this conversation so the user doesn't end
      // up with a permanently-stuck bubble in the happy-path single-message
      // case.
      let consumed: PendingUserMessage | null = null;
      set((state) => {
        const sending = state.pendingMessages
          .map((m, i) => ({ m, i }))
          .filter(
            ({ m }) =>
              m.status === "sending" && m.conversationId === conversationId,
          );
        if (sending.length === 0) return state;
        const exact = sending.find(({ m }) => m.content === content);
        const target = exact ?? sending[0];
        consumed = target.m;
        return {
          pendingMessages: [
            ...state.pendingMessages.slice(0, target.i),
            ...state.pendingMessages.slice(target.i + 1),
          ],
        };
      });
      return consumed;
    },

    clearPendingMessages: () => set(() => ({ ...initialState })),

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
