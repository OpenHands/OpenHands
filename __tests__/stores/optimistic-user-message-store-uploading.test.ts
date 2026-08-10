/**
 * Tests for the new "uploading" lifecycle added in Issue #16430.
 * Run in isolation: `npx vitest run __tests__/stores/optimistic-user-message-store-uploading`
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PENDING_MESSAGE_TIMEOUT_MS,
  useOptimisticUserMessageStore,
} from "#/stores/optimistic-user-message-store";

const CONVO = "conv-upload";

describe("optimistic-user-message-store — uploading lifecycle (#16430)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("enqueues with status 'uploading' and initial uploadProgress", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({
      conversationId: CONVO,
      text: "attach file",
      status: "uploading",
      uploadProgress: 0,
    });
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.id).toBe(id);
    expect(msg.status).toBe("uploading");
    expect(msg.uploadProgress).toBe(0);
  });

  it("still defaults to status 'sending' when no status provided", () => {
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({ conversationId: CONVO, text: "plain" });
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.status).toBe("sending");
    expect(msg.uploadProgress).toBeUndefined();
  });

  it("does NOT arm watchdog for uploading messages", () => {
    const store = useOptimisticUserMessageStore.getState();
    store.enqueuePendingMessage({ conversationId: CONVO, text: "big", status: "uploading", uploadProgress: 0 });
    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS + 1000);
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.status).toBe("uploading");
  });

  it("updatePendingMessageProgress sets uploadProgress", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({ conversationId: CONVO, text: "file", status: "uploading", uploadProgress: 0 });
    store.updatePendingMessageProgress(id, 45);
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.uploadProgress).toBe(45);
  });

  it("updatePendingMessageProgress does not mutate other messages", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id1 = store.enqueuePendingMessage({ conversationId: CONVO, text: "f1", status: "uploading", uploadProgress: 0 });
    const id2 = store.enqueuePendingMessage({ conversationId: CONVO, text: "f2", status: "uploading", uploadProgress: 0 });
    store.updatePendingMessageProgress(id1, 60);
    const msgs = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msgs.find((m) => m.id === id1)!.uploadProgress).toBe(60);
    expect(msgs.find((m) => m.id === id2)!.uploadProgress).toBe(0);
  });

  it("updatePendingMessage transitions uploading -> sending and clears progress", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({ conversationId: CONVO, text: "attach", status: "uploading", uploadProgress: 0 });
    store.updatePendingMessage(id, { content: "attach\n\nFiles: foo.txt", fileUrls: ["foo.txt"], status: "sending", uploadProgress: undefined });
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.status).toBe("sending");
    expect(msg.content).toBe("attach\n\nFiles: foo.txt");
    expect(msg.fileUrls).toEqual(["foo.txt"]);
    expect(msg.uploadProgress).toBeUndefined();
  });

  it("updatePendingMessage arms the watchdog when transitioning to sending", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({ conversationId: CONVO, text: "big", status: "uploading", uploadProgress: 0 });
    store.updatePendingMessage(id, { status: "sending" });
    vi.advanceTimersByTime(PENDING_MESSAGE_TIMEOUT_MS + 1000);
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.status).toBe("error");
    expect(msg.errorMessage).toBe("Send timed out");
  });

  it("markPendingMessageError works on uploading messages", () => {
    const store = useOptimisticUserMessageStore.getState();
    const id = store.enqueuePendingMessage({ conversationId: CONVO, text: "fail", status: "uploading", uploadProgress: 20 });
    store.markPendingMessageError(id, "Network error during upload");
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.status).toBe("error");
    expect(msg.errorMessage).toBe("Network error during upload");
  });

  it("updatePendingMessage preserves unaffected fields", () => {
    const store = useOptimisticUserMessageStore.getState();
    const ts = "2025-01-01T00:00:00.000Z";
    const id = store.enqueuePendingMessage({ conversationId: CONVO, text: "hello", status: "uploading", uploadProgress: 0, timestamp: ts, imageUrls: ["img.png"] });
    store.updatePendingMessage(id, { status: "sending" });
    const [msg] = useOptimisticUserMessageStore.getState().pendingMessages;
    expect(msg.text).toBe("hello");
    expect(msg.timestamp).toBe(ts);
    expect(msg.imageUrls).toEqual(["img.png"]);
  });
});
