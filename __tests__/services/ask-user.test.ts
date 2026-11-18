import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ASK_USER_ACTION_KIND,
  ASK_USER_RESULT_PREFIX,
} from "#/constants/ask-user";
import {
  completeAskUserDecision,
  handleAskUserAction,
} from "#/services/ask-user";
import {
  type AskUserDecisionRequest,
  useAskUserStore,
} from "#/stores/ask-user-store";
import type { AskUserAction } from "#/types/agent-server/core";

const { mockSendMessage } = vi.hoisted(() => ({
  mockSendMessage: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { sendMessage: mockSendMessage } }),
);

const CONVERSATION_ID = "conversation-1";

const action: AskUserAction = {
  kind: ASK_USER_ACTION_KIND,
  question: "Which test runner should we configure?",
  options: [
    { label: "Vitest", description: "Fast Vite-native test runner" },
    { label: "Jest", description: "Standard test runner" },
  ],
  recommended: "Vitest",
};

describe("ask-user service", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAskUserStore.getState().clear();
    mockSendMessage.mockReset();
  });

  it("claims the tool call and enqueues a decision request", async () => {
    await handleAskUserAction(action, CONVERSATION_ID, "tool-call-1");

    const requests = useAskUserStore.getState().requests;
    expect(requests).toHaveLength(1);
    expect(requests[0]).toEqual({
      conversationId: CONVERSATION_ID,
      toolCallId: "tool-call-1",
      question: "Which test runner should we configure?",
      options: [
        { label: "Vitest", description: "Fast Vite-native test runner" },
        { label: "Jest", description: "Standard test runner" },
      ],
      recommended: "Vitest",
      selectedAnswer: null,
    });
  });

  it("ignores replayed ActionEvents with the same toolCallId", async () => {
    await handleAskUserAction(action, CONVERSATION_ID, "tool-call-1");
    await handleAskUserAction(action, CONVERSATION_ID, "tool-call-1");

    expect(useAskUserStore.getState().requests).toHaveLength(1);
  });

  it("completes decision by updating store and posting client tool result back", async () => {
    const request: AskUserDecisionRequest = {
      conversationId: CONVERSATION_ID,
      toolCallId: "tool-call-1",
      question: action.question,
      options: action.options,
      recommended: action.recommended,
      selectedAnswer: null,
    };
    useAskUserStore.getState().enqueue(request);

    await completeAskUserDecision(request, "Vitest");

    // Store is marked as answered
    const updated = useAskUserStore
      .getState()
      .requests.find((r) => r.toolCallId === "tool-call-1");
    expect(updated?.selectedAnswer).toBe("Vitest");

    // Reported back to agent server
    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    const [conversationId, message] = mockSendMessage.mock.calls[0] as [
      string,
      { content: { text: string }[] },
    ];
    expect(conversationId).toBe(CONVERSATION_ID);
    expect(message.content[0].text.startsWith(ASK_USER_RESULT_PREFIX)).toBe(
      true,
    );

    const payload = JSON.parse(
      message.content[0].text.slice(ASK_USER_RESULT_PREFIX.length),
    );
    expect(payload).toEqual({
      toolCallId: "tool-call-1",
      answer: "Vitest",
    });
  });
});
