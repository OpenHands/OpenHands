import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PENDING_MESSAGE_TIMEOUT_MS,
  useOptimisticUserMessageStore,
} from "#/stores/optimistic-user-message-store";
import {
  linkPendingTaskMessages,
  resetPendingTaskMessageLinkState,
} from "#/utils/pending-task-message-link";

import { useEventStore } from "#/stores/use-event-store";

const CONVO = "conv-a";

function addServerMessage(id: string, timestamp: string) {
  useEventStore.getState().addEvent({
    id,
    timestamp,
    source: "user",
    llm_message: {
      role: "user",
      content: [{ type: "text", text: "continue" }],
    },
    activated_skills: [],
    extended_content: [],
  });
}

describe("optimistic-user-message-store", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useOptimisticUserMessageStore.getState().clearPendingMessages();
    resetPendingTaskMessageLinkState();
    useEventStore.getState().clearEventsForConversation(CONVO);
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
    const otherId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "other",
    });
    store.markPendingMessageError(id, "boom");
    store.markPendingMessageError(otherId, "other boom");

    store.markPendingMessageSending(id);

    const [entry, other] =
      useOptimisticUserMessageStore.getState().pendingMessages;
    expect(entry.status).toBe("sending");
    expect(entry.errorMessage).toBeUndefined();
    expect(other.status).toBe("error");
    expect(other.errorMessage).toBe("other boom");
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
    const consumed = store.consumeMatchingPendingMessage(CONVO, "second", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

    expect(consumed?.id).toBe(secondId);
    const remaining = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(firstId);
  });

  it("consumeMatchingPendingMessage matches trimmed content without consuming unrelated sending entries", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "hello",
    });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "world" });

    const consumed = store.consumeMatchingPendingMessage(CONVO, " hello ", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

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

    const consumed = store.consumeMatchingPendingMessage(CONVO, "second", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

    expect(consumed?.id).toBe(secondId);
    const remaining = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(firstId);
    expect(remaining[0].status).toBe("error");
  });

  it("consumeMatchingPendingMessage is a no-op when only errored entries exist and none match exactly", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "broken",
    });
    store.markPendingMessageError(id, "boom");

    const consumed = store.consumeMatchingPendingMessage(CONVO, "unrelated", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

    expect(consumed).toBeNull();
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(1);
  });

  it("consumeMatchingPendingMessage clears an errored entry when the echo matches its content exactly", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "late echo",
    });
    // The watchdog gives up before the server echoes the message back.
    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS);
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages[0].status,
    ).toBe("error");

    const consumed = store.consumeMatchingPendingMessage(CONVO, "late echo", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

    expect(consumed?.id).toBe(id);
    expect(
      useOptimisticUserMessageStore.getState().pendingMessages,
    ).toHaveLength(0);
  });

  it.each(["sending", "error"] as const)(
    "keeps a newer %s message when identical old history arrives",
    (status) => {
      const store = useOptimisticUserMessageStore.getState();
      const oldTimestamp = new Date(Date.now() - 1000).toISOString();
      addServerMessage("old", oldTimestamp);
      const id = store.enqueuePendingMessage({
        conversationId: CONVO,
        text: "continue",
      });
      if (status === "error") store.markPendingMessageError(id, "Rejected");

      store.consumeMatchingPendingMessage(CONVO, "continue", {
        id: "old",
        timestamp: oldTimestamp,
      });

      expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual([
        expect.objectContaining({ id, status }),
      ]);
    },
  );

  it.each([-60_000, 60_000])(
    "uses server history when the browser clock differs by %d ms",
    (clockOffset) => {
      const serverTime = Date.parse("2026-09-24T12:00:00Z");
      vi.setSystemTime(serverTime + clockOffset);
      addServerMessage("old", new Date(serverTime - 1000).toISOString());
      const store = useOptimisticUserMessageStore.getState();
      const id = store.enqueuePendingMessage({
        conversationId: CONVO,
        text: "continue",
      });

      expect(
        store.consumeMatchingPendingMessage(CONVO, "continue", {
          id: "old",
          timestamp: new Date(serverTime - 1000).toISOString(),
        }),
      ).toBeNull();
      expect(
        store.consumeMatchingPendingMessage(CONVO, "continue", {
          id: "new",
          timestamp: new Date(serverTime).toISOString(),
        })?.id,
      ).toBe(id);
    },
  );

  it("confirms the sending attempt before an older identical failed attempt", () => {
    const store = useOptimisticUserMessageStore.getState();
    const failed = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "continue",
    });
    store.markPendingMessageError(failed, "Rejected");
    const sending = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "continue",
    });

    const confirmed = store.consumeMatchingPendingMessage(CONVO, "continue", {
      id: "accepted",
      timestamp: new Date().toISOString(),
    });

    expect(confirmed?.id).toBe(sending);
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual([
      expect.objectContaining({ id: failed, status: "error" }),
    ]);
  });

  it("refreshes the server history boundary when retrying a failed attempt", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "continue",
    });
    store.markPendingMessageError(id, "Rejected");
    const oldTimestamp = new Date().toISOString();
    addServerMessage("other-attempt", oldTimestamp);

    store.markPendingMessageSending(id);

    expect(
      store.consumeMatchingPendingMessage(CONVO, "continue", {
        id: "other-attempt",
        timestamp: oldTimestamp,
      }),
    ).toBeNull();
    expect(
      store.consumeMatchingPendingMessage(CONVO, "continue", {
        id: "retry-accepted",
        timestamp: new Date(Date.now() + 1000).toISOString(),
      })?.id,
    ).toBe(id);
  });

  it("does not confirm two identical attempts with a replayed event", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: CONVO, text: "continue" });
    const failed = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "continue",
    });
    store.markPendingMessageError(failed, "Rejected");
    const event = { id: "accepted", timestamp: new Date().toISOString() };

    store.consumeMatchingPendingMessage(CONVO, "continue", event);
    store.consumeMatchingPendingMessage(CONVO, "continue", event);

    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual([
      expect.objectContaining({ id: failed, status: "error" }),
    ]);
  });

  it("retains a sending message for an unrelated newer confirmation", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "continue",
    });
    store.consumeMatchingPendingMessage(CONVO, "different message", {
      id: "other",
      timestamp: new Date().toISOString(),
    });
    expect(useOptimisticUserMessageStore.getState().pendingMessages[0].id).toBe(
      id,
    );
  });

  it("confirms a task's first message before the pending route reassignment", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: "task-123", text: "hello" });
    linkPendingTaskMessages(CONVO, "task-123");
    store.consumeMatchingPendingMessage(CONVO, "hello", {
      id: "first",
      timestamp: new Date().toISOString().replace("Z", ""),
    });
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toEqual(
      [],
    );
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
    const consumed = store.consumeMatchingPendingMessage("conv-b", "shared", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });

    expect(consumed?.id).toBe(bId);
    const remaining = useOptimisticUserMessageStore.getState().pendingMessages;
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

    store.consumeMatchingPendingMessage(CONVO, "fast", {
      id: "confirmation",
      timestamp: new Date().toISOString(),
    });
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
    const waitingId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "still-waiting",
    });
    store.markPendingMessageError(id, "boom");

    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS);

    const [entry, waiting] =
      useOptimisticUserMessageStore.getState().pendingMessages;
    // Should keep the original error message, not get overwritten to "Send timed out".
    expect(entry.errorMessage).toBe("boom");
    expect(waiting.id).toBe(waitingId);
    expect(waiting.status).toBe("error");
    expect(waiting.errorMessage).toBe("Send timed out");
  });

  it("removePendingMessage drops a specific entry by id", () => {
    const store = useOptimisticUserMessageStore.getState();
    const firstId = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "first",
    });
    store.enqueuePendingMessage({ conversationId: CONVO, text: "second" });

    store.removePendingMessage(firstId);

    const remaining = useOptimisticUserMessageStore.getState().pendingMessages;
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

  it("creates a complete fresh store whose actions remain scoped", async () => {
    const randomSpy = vi
      .spyOn(Math, "random")
      .mockReturnValueOnce(0.5)
      .mockReturnValue(0.25);
    vi.resetModules();

    try {
      const { useOptimisticUserMessageStore: freshStore } =
        await import("#/stores/optimistic-user-message-store");
      expect(freshStore.getState().pendingMessages).toEqual([]);

      const firstId = freshStore.getState().enqueuePendingMessage({
        conversationId: CONVO,
        text: "first",
      });
      const secondId = freshStore.getState().enqueuePendingMessage({
        conversationId: "conv-b",
        text: "second",
      });
      expect(typeof firstId).toBe("string");
      expect(typeof secondId).toBe("string");
      expect(firstId).not.toBe(secondId);

      freshStore.getState().markPendingMessageError(firstId, "failed");
      expect(
        freshStore
          .getState()
          .pendingMessages.map(({ text, status, errorMessage }) => [
            text,
            status,
            errorMessage,
          ]),
      ).toEqual([
        ["first", "error", "failed"],
        ["second", "sending", undefined],
      ]);

      freshStore.getState().markPendingMessageSending(firstId);
      expect(freshStore.getState().pendingMessages[0]).toMatchObject({
        id: firstId,
        status: "sending",
        errorMessage: undefined,
      });

      freshStore.getState().reassignPendingMessages("conv-b", "moved");
      expect(
        freshStore
          .getState()
          .pendingMessages.map(({ conversationId }) => conversationId),
      ).toEqual([CONVO, "moved"]);

      freshStore.getState().removePendingMessage(firstId);
      expect(freshStore.getState().pendingMessages).toHaveLength(1);
      expect(freshStore.getState().pendingMessages[0].id).toBe(secondId);

      freshStore.getState().clearPendingMessages();
      expect(freshStore.getState().pendingMessages).toEqual([]);
    } finally {
      randomSpy.mockRestore();
      vi.resetModules();
    }
  });
});
