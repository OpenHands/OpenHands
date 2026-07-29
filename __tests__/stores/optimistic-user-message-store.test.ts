import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PENDING_MESSAGE_TIMEOUT_MS,
  useOptimisticUserMessageStore,
} from "#/stores/optimistic-user-message-store";

const CONVO = "conv-a";

describe("optimistic-user-message-store", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("enqueues new messages with status 'sending' and tags them with conversationId", () => {
    const store = useOptimisticUserMessageStore.getState();

    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "hello",
    });

    const pending = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(pending).toHaveLength(1);
    expect(pending[0].id).toBe(id);
    expect(pending[0].conversationId).toBe(CONVO);
    expect(pending[0].text).toBe("hello");
    expect(pending[0].status).toBe("sending");
    expect(pending[0].imageUrls).toEqual([]);
    expect(pending[0].fileUrls).toEqual([]);
    expect(typeof pending[0].timestamp).toBe("string");
  });

  it("preserves FIFO order across multiple enqueues", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: CONVO, text: "first" });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "second" });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "third" });

    const pending = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(pending.map((m) => m.text)).toEqual(["first", "second", "third"]);
  });

  it("marks a pending message as 'error' with details", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "broken",
    });

    store.markPendingMessageError(id, "boom");

    const [entry] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(entry.status).toBe("error");
    expect(entry.errorMessage).toBe("boom");
  });

  it("flips an errored message back to 'sending' on retry", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "broken",
    });
    store.markPendingMessageError(id, "boom");

    store.markPendingMessageSending(id);

    const [entry] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(entry.status).toBe("sending");
    expect(entry.errorMessage).toBeUndefined();
  });

  it("enqueue stores `content` separately from `text` and defaults it to `text`", () => {
    const store = useOptimisticUserMessageStore.getState();
    const idA = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "hello",
    });
    const idB = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "hello",
      content: "hello\n\nFiles: foo.txt",
    });

    const pending = useOptimisticUserMessageStore.getState().pendingMessages;
    const a = pending.find((m) => m.id === idA)!;
    const b = pending.find((m) => m.id === idB)!;
    expect(a.content).toBe("hello");
    expect(b.text).toBe("hello");
    expect(b.content).toBe("hello\n\nFiles: foo.txt");
  });

  it("consumeMatchingPendingMessage prefers an exact content match (out-of-order echo)", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "first",
    });
    const secondId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "second",
    });

    // Echo for "second" arrives before "first" — must pop "second", not the
    // oldest entry. This is the case the previous FIFO-only implementation
    // got wrong.
    const consumed = store.consumeMatchingPendingMessage(CONVO, "second");

    expect(consumed?.id).toBe(secondId);
    const remaining =
      useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(firstId);
  });

  it("consumeMatchingPendingMessage falls back to oldest sending entry when no exact match exists", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "hello",
    });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "world" });

    // Server munged the echo (e.g., trimmed whitespace). FIFO fallback keeps
    // the bubble from getting stuck.
    const consumed = store.consumeMatchingPendingMessage(
      CONVO,
      "something else",
    );

    expect(consumed?.id).toBe(firstId);
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(1);
  });

  it("consumeMatchingPendingMessage skips entries already in 'error' state", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "first",
    });
    const secondId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "second",
    });
    store.markPendingMessageError(firstId, "boom");

    const consumed = store.consumeMatchingPendingMessage(CONVO, "second");

    expect(consumed?.id).toBe(secondId);
    const remaining =
      useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(firstId);
    expect(remaining[0].status).toBe("error");
  });

  it("consumeMatchingPendingMessage is a no-op when nothing is sending", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "broken",
    });
    store.markPendingMessageError(id, "boom");

    const consumed = store.consumeMatchingPendingMessage(CONVO, "broken");

    expect(consumed).toBeNull();
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(1);
  });

  it("consumeMatchingPendingMessage only consumes entries for the given conversation", () => {
    const store = useOptimisticUserMessageStore.getState();
    const aId = store.enqueuePendingMessage({
      conversationId: "conv-a",
      text: "shared",
    });
    const bId = store.enqueuePendingMessage({
      conversationId: "conv-b",
      text: "shared",
    });

    // A cross-conversation ack for conv-b — even with identical content,
    // must not pop conv-a's pending entry.
    const consumed = store.consumeMatchingPendingMessage("conv-b", "shared");

    expect(consumed?.id).toBe(bId);
    const remaining =
      useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(aId);
  });

  it("enqueuePendingMessage flips the entry to 'error' after the watchdog timeout", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "stuck",
    });

    // Still sending right after enqueue.
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages[0].status,
    ).toBe("sending");

    // Fire the watchdog.
    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS);

    const [entry] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(entry.id).toBe(id);
    expect(entry.status).toBe("error");
    expect(entry.errorMessage).toBe("Send timed out");
  });

  it("watchdog timeout does nothing if the echo already consumed the message", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: CONVO, text: "fast" });

    store.consumeMatchingPendingMessage(CONVO, "fast");
    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS);

    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(0);
  });

  it("watchdog timeout does nothing if the message already failed via send error", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "explicit-error",
    });
    store.markPendingMessageError(id, "boom");

    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS);

    const [entry] = useOptimisticUserMessageStore.getState().pendingMessages;
    // Should keep the original error message, not get overwritten to "Send timed out".
    expect(entry.errorMessage).toBe("boom");
  });

  it("removePendingMessage drops a specific entry by id", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "first",
    });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "second" });

    store.removePendingMessage(firstId);

    const remaining =
      useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining.map((m) => m.text)).toEqual(["second"]);
  });

  it("clearPendingMessages wipes the queue", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: CONVO, text: "first" });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "second" });

    store.clearPendingMessages();

    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(0);
  });

  it("reassignPendingMessages moves entries from a task id to the real conversation id", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({
      conversationId: "task-abc",
      text: "hello",
    });
    store.enqueuePendingMessage({
      conversationId: "other-convo",
      text: "untouched",
    });

    store.reassignPendingMessages("task-abc", "real-convo");

    const pending = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(pending.map((m) => [m.conversationId, m.text])).toEqual([
      ["real-convo", "hello"],
      ["other-convo", "untouched"],
    ]);
  });
});
