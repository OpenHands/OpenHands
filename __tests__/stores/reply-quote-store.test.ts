import { beforeEach, describe, expect, it } from "vitest";
import type { ReplyQuote } from "#/utils/message-quote";
import {
  useReplyQuoteStore,
  type ArmedReplyQuote,
} from "#/stores/reply-quote-store";
import {
  getConversationState,
  LOCAL_STORAGE_KEYS,
  setConversationState,
} from "#/utils/conversation-local-storage";

const CONVO_A = "conv-quote-a";
const CONVO_B = "conv-quote-b";

function makeQuote(overrides: Partial<ReplyQuote> = {}): ReplyQuote {
  return {
    sourceEventId: "evt-source-1",
    source: "agent",
    body: "the original question",
    truncated: false,
    ...overrides,
  };
}

function resetStore() {
  useReplyQuoteStore.setState({ pendingReplyQuote: null });
}

// Test-only render of the hook behavior: zustand selectors are plain
// functions over store state, so assert against store state plus the same
// conversation filter the selector applies (the hook itself is exercised in
// component tests in later tasks).
function selectPendingQuote(conversationId: string): ArmedReplyQuote | null {
  const armed = useReplyQuoteStore.getState().pendingReplyQuote;
  return armed?.conversationId === conversationId ? armed : null;
}

describe("reply-quote-store", () => {
  beforeEach(() => {
    localStorage.clear();
    resetStore();
  });

  it("setReplyQuote arms the quote for the conversation and tags it", () => {
    // Act
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, makeQuote());

    // Assert
    const armed = useReplyQuoteStore.getState().pendingReplyQuote;
    expect(armed).toEqual({ ...makeQuote(), conversationId: CONVO_A });
    expect(selectPendingQuote(CONVO_A)).toEqual(armed);
  });

  it("an armed quote in one conversation does not leak into another", () => {
    // Arrange
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, makeQuote());

    // Assert
    expect(selectPendingQuote(CONVO_B)).toBeNull();
  });

  it("setReplyQuote persists the quote to the conversation localStorage", () => {
    // Act
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, makeQuote());

    // Assert
    const stored = getConversationState(CONVO_A).pendingReplyQuote;
    expect(stored).toEqual(makeQuote());

    const raw = JSON.parse(
      localStorage.getItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${CONVO_A}`,
      )!,
    );
    expect(raw.pendingReplyQuote).toEqual(makeQuote());
  });

  it("clearReplyQuote removes the armed quote and persists null", () => {
    // Arrange
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, makeQuote());

    // Act
    useReplyQuoteStore.getState().clearReplyQuote(CONVO_A);

    // Assert
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
    expect(getConversationState(CONVO_A).pendingReplyQuote).toBeNull();
  });

  it("clearReplyQuote skips persistence writes for task conversation ids", () => {
    // Arrange
    const taskId = "task-quote-1";
    useReplyQuoteStore.getState().setReplyQuote(taskId, makeQuote());
    expect(getConversationState(taskId).pendingReplyQuote).toBeNull();

    // Act — clearing must not throw and must clear in-memory state
    useReplyQuoteStore.getState().clearReplyQuote(taskId);

    // Assert
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
  });

  it("hydrateReplyQuote restores a persisted quote for its conversation", () => {
    // Arrange — the quote was armed, persisted, and the SPA reloaded
    setConversationState(CONVO_A, { pendingReplyQuote: makeQuote() });
    resetStore();

    // Act
    useReplyQuoteStore.getState().hydrateReplyQuote(CONVO_A);

    // Assert
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toEqual({
      ...makeQuote(),
      conversationId: CONVO_A,
    });
  });

  it("hydrateReplyQuote clears a stale armed quote when switching conversations", () => {
    // Arrange — armed in conv A mid-compose, then the user switches to conv B
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, makeQuote());

    // Act
    useReplyQuoteStore.getState().hydrateReplyQuote(CONVO_B);

    // Assert — conv B has no persisted quote, so nothing is armed
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
  });

  it("hydrateReplyQuote keeps an already-armed quote for the same conversation", () => {
    // Arrange
    const freshQuote = makeQuote({ sourceEventId: "evt-newer" });
    useReplyQuoteStore.getState().setReplyQuote(CONVO_A, freshQuote);

    // Act
    useReplyQuoteStore.getState().hydrateReplyQuote(CONVO_A);

    // Assert
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toEqual({
      ...freshQuote,
      conversationId: CONVO_A,
    });
  });
});
