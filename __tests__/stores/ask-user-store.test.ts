import { beforeEach, describe, expect, it } from "vitest";
import {
  type AskUserDecisionRequest,
  useAskUserStore,
} from "#/stores/ask-user-store";

describe("ask-user-store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAskUserStore.getState().clear();
  });

  it("enqueues a decision request and avoids duplicates", () => {
    const request: AskUserDecisionRequest = {
      conversationId: "conv-1",
      toolCallId: "call-1",
      question: "Which database adapter should we configure?",
      options: [
        { label: "PostgreSQL", description: "Standard relational DB" },
        { label: "SQLite", description: "In-memory or single file" },
      ],
      recommended: "PostgreSQL",
      selectedAnswer: null,
    };

    useAskUserStore.getState().enqueue(request);
    expect(useAskUserStore.getState().requests).toHaveLength(1);
    expect(useAskUserStore.getState().requests[0]).toEqual(request);

    // Re-enqueuing with the same toolCallId is a no-op
    useAskUserStore.getState().enqueue(request);
    expect(useAskUserStore.getState().requests).toHaveLength(1);
  });

  it("records the user's selected answer", () => {
    const request: AskUserDecisionRequest = {
      conversationId: "conv-1",
      toolCallId: "call-2",
      question: "Should we run tests with coverage?",
      options: [{ label: "Yes" }, { label: "No" }],
      selectedAnswer: null,
    };

    useAskUserStore.getState().enqueue(request);
    useAskUserStore.getState().answer("call-2", "Yes");

    const updated = useAskUserStore
      .getState()
      .requests.find((r) => r.toolCallId === "call-2");
    expect(updated?.selectedAnswer).toBe("Yes");
  });

  it("dismisses a request by toolCallId", () => {
    const req1: AskUserDecisionRequest = {
      conversationId: "conv-1",
      toolCallId: "call-1",
      question: "Q1",
      options: [],
    };
    const req2: AskUserDecisionRequest = {
      conversationId: "conv-1",
      toolCallId: "call-2",
      question: "Q2",
      options: [],
    };

    useAskUserStore.getState().enqueue(req1);
    useAskUserStore.getState().enqueue(req2);
    expect(useAskUserStore.getState().requests).toHaveLength(2);

    useAskUserStore.getState().dismiss("call-1");
    expect(useAskUserStore.getState().requests).toHaveLength(1);
    expect(useAskUserStore.getState().requests[0].toolCallId).toBe("call-2");
  });
});
