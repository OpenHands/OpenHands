import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  REQUEST_SECRET_ACTION_KIND,
  SECRET_REQUEST_RESULT_PREFIX,
} from "#/constants/secret-request";
import {
  completeSecretRequest,
  handleRequestSecretAction,
} from "#/services/secret-request";
import { useGoalStore } from "#/stores/goal-store";
import {
  type SecretRequest,
  useSecretRequestStore,
} from "#/stores/secret-request-store";
import type { RequestSecretAction } from "#/types/agent-server/core";

const { mockSendMessage, mockAttachConversationSecret, mockDisplayErrorToast } =
  vi.hoisted(() => ({
    mockSendMessage: vi.fn(),
    mockAttachConversationSecret: vi.fn(),
    mockDisplayErrorToast: vi.fn(),
  }));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mockDisplayErrorToast,
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { sendMessage: mockSendMessage } }),
);

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  attachConversationSecret: mockAttachConversationSecret,
}));

const CONVERSATION_ID = "conversation-1";

const action: RequestSecretAction = {
  kind: REQUEST_SECRET_ACTION_KIND,
  name: "GITHUB_TOKEN",
  reason: "Push the factory branch; needs repo scope.",
};

const request: SecretRequest = {
  conversationId: CONVERSATION_ID,
  toolCallId: "tool-call-1",
  name: "GITHUB_TOKEN",
  reason: "Push the factory branch; needs repo scope.",
};

/** The text the service posted back into the conversation. */
function reportedText(): string {
  const [conversationId, message] = mockSendMessage.mock.calls.at(-1) as [
    string,
    { content: { text: string }[] },
  ];
  expect(conversationId).toBe(CONVERSATION_ID);
  return message.content[0].text;
}

function reportedResult() {
  const text = reportedText();
  expect(text.startsWith(SECRET_REQUEST_RESULT_PREFIX)).toBe(true);
  return JSON.parse(text.slice(SECRET_REQUEST_RESULT_PREFIX.length));
}

beforeEach(() => {
  window.localStorage.clear();
  useGoalStore.setState({ statusByConversation: {} });
  useSecretRequestStore.setState({ requests: [request] });
  vi.clearAllMocks();
  mockSendMessage.mockResolvedValue(undefined);
  mockAttachConversationSecret.mockResolvedValue({ status: "attached" });
});

describe("handleRequestSecretAction", () => {
  it("queues one request per tool call, ignoring replays", async () => {
    useSecretRequestStore.setState({ requests: [] });
    await handleRequestSecretAction(action, CONVERSATION_ID, "tool-call-1");
    await handleRequestSecretAction(action, CONVERSATION_ID, "tool-call-1");

    expect(useSecretRequestStore.getState().requests).toEqual([request]);
    expect(mockAttachConversationSecret).not.toHaveBeenCalled();
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("completeSecretRequest", () => {
  it("attaches a saved secret and reports its name, never its value", async () => {
    await completeSecretRequest(request, {
      name: "GH_FACTORY_TOKEN",
      description: "Factory repo access",
    });

    expect(useSecretRequestStore.getState().requests).toEqual([]);
    expect(mockAttachConversationSecret).toHaveBeenCalledExactlyOnceWith(
      CONVERSATION_ID,
      { name: "GH_FACTORY_TOKEN", description: "Factory repo access" },
    );
    expect(reportedResult()).toEqual({
      status: "saved",
      name: "GH_FACTORY_TOKEN",
    });
  });

  it("reports a closed form as not saved without attaching anything", async () => {
    await completeSecretRequest(request, null);

    expect(mockAttachConversationSecret).not.toHaveBeenCalled();
    expect(reportedResult()).toMatchObject({
      status: "not_saved",
      name: "GITHUB_TOKEN",
    });
  });

  it("reports an error when the saved secret cannot reach the conversation", async () => {
    mockAttachConversationSecret.mockRejectedValueOnce(
      new Error("Conversation not found"),
    );

    await completeSecretRequest(request, { name: "GITHUB_TOKEN" });

    expect(reportedResult()).toMatchObject({
      status: "error",
      name: "GITHUB_TOKEN",
      error: "Conversation not found",
    });
    expect(mockDisplayErrorToast).toHaveBeenCalledWith(
      "Conversation not found",
    );
  });

  it("reports a secret the sandbox could not take yet as not saved", async () => {
    mockAttachConversationSecret.mockResolvedValueOnce({
      status: "deferred",
      message: "Sandbox is PAUSED",
    });

    await completeSecretRequest(request, { name: "GITHUB_TOKEN" });

    const result = reportedResult();
    expect(result).toMatchObject({ status: "not_saved", name: "GITHUB_TOKEN" });
    expect(result.guidance).toContain("not attached to this conversation");
    expect(mockDisplayErrorToast).not.toHaveBeenCalled();
  });

  it.each([
    [
      "not_found",
      "The saved secret GITHUB_TOKEN could not be found.",
      "Settings > Secrets",
    ],
    ["error", "Agent server error: 500", "available in new conversations"],
  ])(
    "reports an attach answered %s as an error",
    async (status, error, guidance) => {
      mockAttachConversationSecret.mockResolvedValueOnce({
        status,
        message: status === "error" ? "Agent server error: 500" : "ignored",
      });

      await completeSecretRequest(request, { name: "GITHUB_TOKEN" });

      const result = reportedResult();
      expect(result).toMatchObject({
        status: "error",
        name: "GITHUB_TOKEN",
        error,
      });
      expect(result.guidance).toContain(guidance);
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(error);
    },
  );

  it("reports a request only once when the form settles after a cancel", async () => {
    await completeSecretRequest(request, null);
    await completeSecretRequest(request, { name: "GITHUB_TOKEN" });

    expect(mockAttachConversationSecret).not.toHaveBeenCalled();
    expect(mockSendMessage).toHaveBeenCalledOnce();
    expect(reportedResult()).toMatchObject({ status: "not_saved" });
  });

  // The agent server cancels an active /goal loop on any inbound message.
  it("does not post the result while a goal loop is running", async () => {
    useGoalStore.setState({
      statusByConversation: {
        [CONVERSATION_ID]: {
          active: true,
          status: "running",
          iteration: 1,
          max_iterations: 5,
          objective: "ship it",
          verdict: null,
        },
      },
    });

    await completeSecretRequest(request, { name: "GITHUB_TOKEN" });

    expect(mockAttachConversationSecret).toHaveBeenCalledOnce();
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});
