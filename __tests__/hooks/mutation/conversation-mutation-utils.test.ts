import type { StartGoalRequest } from "@openhands/typescript-client";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  askAgent,
  invalidateConversationQueries,
  patchConversationInCache,
  pauseConversation,
  resumeConversation,
  resumeGoal,
  startGoal,
  stopGoal,
} from "#/hooks/mutation/conversation-mutation-utils";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const mocks = vi.hoisted(() => ({
  askAgent: vi.fn(),
  batchGetAppConversations: vi.fn(),
  getActiveBackend: vi.fn(),
  getAgentServerClientOptions: vi.fn(),
  interruptConversation: vi.fn(),
  pauseCloudSandbox: vi.fn(),
  resumeGoal: vi.fn(),
  runConversation: vi.fn(),
  startGoal: vi.fn(),
  stopGoal: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(function ConversationClientMock() {
    return {
      askAgent: mocks.askAgent,
      interruptConversation: mocks.interruptConversation,
      resumeGoal: mocks.resumeGoal,
      runConversation: mocks.runConversation,
      startGoal: mocks.startGoal,
      stopGoal: mocks.stopGoal,
    };
  }),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: mocks.getActiveBackend,
}));

vi.mock("#/api/cloud/conversation-service.api", () => ({
  pauseCloudSandbox: mocks.pauseCloudSandbox,
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: mocks.getAgentServerClientOptions,
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      batchGetAppConversations: mocks.batchGetAppConversations,
    },
  }),
);

const CONV_ID = "conv-test-1";
const CLIENT_OPTIONS = {
  host: "https://runtime.example.com",
  apiKey: "runtime-session-key",
  workingDir: "/workspace/project",
};

const makeConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: CONV_ID,
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test conversation",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  execution_status: null,
  conversation_url: "https://runtime.example.com/api/conversations/conv-test-1",
  session_api_key: "runtime-session-key",
  sandbox_id: "sandbox-1",
  sandbox_status: "RUNNING",
  sub_conversation_ids: [],
  ...overrides,
});

const makeQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

const loadFreshConversationMutationUtils = async () => {
  vi.resetModules();
  return import("#/hooks/mutation/conversation-mutation-utils");
};

const prepareRuntimeConversation = (
  overrides: Partial<AppConversation> = {},
) => {
  const conversation = makeConversation(overrides);
  mocks.batchGetAppConversations.mockResolvedValueOnce([conversation]);
  mocks.getAgentServerClientOptions.mockReturnValue(CLIENT_OPTIONS);
  return conversation;
};

const expectRuntimeClient = (
  conversationId: string,
  conversation: AppConversation,
) => {
  expect(mocks.batchGetAppConversations).toHaveBeenCalledOnce();
  expect(mocks.batchGetAppConversations).toHaveBeenCalledWith([conversationId]);
  expect(mocks.getAgentServerClientOptions).toHaveBeenCalledOnce();
  expect(mocks.getAgentServerClientOptions).toHaveBeenCalledWith({
    conversationUrl: conversation.conversation_url,
    sessionApiKey: conversation.session_api_key,
  });
  expect(ConversationClient).toHaveBeenCalledOnce();
  expect(ConversationClient).toHaveBeenCalledWith(CLIENT_OPTIONS);
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("conversation runtime commands", () => {
  it("reports the requested id when the conversation cannot be loaded", async () => {
    mocks.batchGetAppConversations.mockResolvedValueOnce([]);

    await expect(
      askAgent("missing-conversation", "Are you there?"),
    ).rejects.toThrow("V1 conversation not found: missing-conversation");

    expect(mocks.batchGetAppConversations).toHaveBeenCalledWith([
      "missing-conversation",
    ]);
    expect(mocks.getAgentServerClientOptions).not.toHaveBeenCalled();
    expect(ConversationClient).not.toHaveBeenCalled();
    expect(mocks.askAgent).not.toHaveBeenCalled();
  });

  it("interrupts a local conversation through its runtime client", async () => {
    const conversation = prepareRuntimeConversation();
    const response = { success: true };
    mocks.getActiveBackend.mockReturnValueOnce({
      backend: { kind: "local" },
    });
    mocks.interruptConversation.mockResolvedValueOnce(response);

    await expect(pauseConversation(CONV_ID)).resolves.toBe(response);

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.interruptConversation).toHaveBeenCalledOnce();
    expect(mocks.interruptConversation).toHaveBeenCalledWith(CONV_ID);
    expect(mocks.pauseCloudSandbox).not.toHaveBeenCalled();
  });

  it("pauses a cloud sandbox without constructing a runtime client", async () => {
    prepareRuntimeConversation({ sandbox_id: "sandbox-cloud-7" });
    mocks.getActiveBackend.mockReturnValueOnce({
      backend: { kind: "cloud" },
    });
    mocks.pauseCloudSandbox.mockResolvedValueOnce(undefined);

    await expect(pauseConversation(CONV_ID)).resolves.toEqual({
      success: true,
    });

    expect(mocks.pauseCloudSandbox).toHaveBeenCalledOnce();
    expect(mocks.pauseCloudSandbox).toHaveBeenCalledWith("sandbox-cloud-7");
    expect(mocks.getAgentServerClientOptions).not.toHaveBeenCalled();
    expect(ConversationClient).not.toHaveBeenCalled();
    expect(mocks.interruptConversation).not.toHaveBeenCalled();
  });

  it("rejects a cloud pause when the conversation has no sandbox", async () => {
    prepareRuntimeConversation({ sandbox_id: null });
    mocks.getActiveBackend.mockReturnValueOnce({
      backend: { kind: "cloud" },
    });

    await expect(pauseConversation(CONV_ID)).rejects.toThrow(
      `Cannot stop runtime: cloud conversation ${CONV_ID} has no sandbox_id.`,
    );

    expect(mocks.pauseCloudSandbox).not.toHaveBeenCalled();
    expect(mocks.getAgentServerClientOptions).not.toHaveBeenCalled();
    expect(ConversationClient).not.toHaveBeenCalled();
  });

  it("asks the runtime agent the exact side question", async () => {
    const conversation = prepareRuntimeConversation();
    const response = { response: "The build is green." };
    mocks.askAgent.mockResolvedValueOnce(response);

    await expect(askAgent(CONV_ID, "What is the build status?")).resolves.toBe(
      response,
    );

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.askAgent).toHaveBeenCalledOnce();
    expect(mocks.askAgent).toHaveBeenCalledWith(
      CONV_ID,
      "What is the build status?",
    );
  });

  it("starts a goal with the complete request", async () => {
    const conversation = prepareRuntimeConversation();
    const request: StartGoalRequest = {
      objective: "Finish the focused test suite",
      max_iterations: 4,
    };
    mocks.startGoal.mockResolvedValueOnce(undefined);

    await expect(startGoal(CONV_ID, request)).resolves.toBeUndefined();

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.startGoal).toHaveBeenCalledOnce();
    expect(mocks.startGoal).toHaveBeenCalledWith(CONV_ID, request);
  });

  it("stops the active goal for the requested conversation", async () => {
    const conversation = prepareRuntimeConversation();
    mocks.stopGoal.mockResolvedValueOnce(undefined);

    await expect(stopGoal(CONV_ID)).resolves.toBeUndefined();

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.stopGoal).toHaveBeenCalledOnce();
    expect(mocks.stopGoal).toHaveBeenCalledWith(CONV_ID);
  });

  it("resumes the interrupted goal for the requested conversation", async () => {
    const conversation = prepareRuntimeConversation();
    mocks.resumeGoal.mockResolvedValueOnce(undefined);

    await expect(resumeGoal(CONV_ID)).resolves.toBeUndefined();

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.resumeGoal).toHaveBeenCalledOnce();
    expect(mocks.resumeGoal).toHaveBeenCalledWith(CONV_ID);
  });

  it("runs a paused conversation through its runtime client", async () => {
    const conversation = prepareRuntimeConversation();
    const response = { status: "running" };
    mocks.runConversation.mockResolvedValueOnce(response);

    await expect(resumeConversation(CONV_ID)).resolves.toBe(response);

    expectRuntimeClient(CONV_ID, conversation);
    expect(mocks.runConversation).toHaveBeenCalledOnce();
    expect(mocks.runConversation).toHaveBeenCalledWith(CONV_ID);
  });
});

describe("conversation cache synchronization", () => {
  it("patches the matching single-item cache across backend key variants", () => {
    const queryClient = makeQueryClient();
    const queryKey = ["user", "conversation", CONV_ID, "backend-local", null];
    const conversation = makeConversation({ sandbox_status: "RUNNING" });
    queryClient.setQueryData(queryKey, conversation);

    patchConversationInCache(queryClient, CONV_ID, {
      sandbox_status: "PAUSED",
    });

    expect(queryClient.getQueryData(queryKey)).toEqual({
      ...conversation,
      sandbox_status: "PAUSED",
    });
  });

  it("does not alter unrelated query families", () => {
    const queryClient = makeQueryClient();
    const unrelatedKey = ["settings", "conversation"];
    const unrelatedData = {
      activeProfile: "default",
      confirmationPolicy: "never",
    };
    queryClient.setQueryData(unrelatedKey, unrelatedData);

    patchConversationInCache(queryClient, CONV_ID, {
      sandbox_status: "PAUSED",
    });

    expect(queryClient.getQueryData(unrelatedKey)).toBe(unrelatedData);
  });

  it("patches every matching list item while preserving pages and other conversations", () => {
    const queryClient = makeQueryClient();
    const queryKey = ["user", "conversations", "backend-local", null];
    const firstMatch = makeConversation({ title: "First match" });
    const secondMatch = makeConversation({ title: "Second match" });
    const other = makeConversation({ id: "conv-other", title: "Other" });
    const firstPage = { items: [firstMatch, other], next_page_id: "page-2" };
    const secondPage = { items: [secondMatch], next_page_id: null };
    queryClient.setQueryData(queryKey, {
      pages: [firstPage, secondPage],
      pageParams: [null, "page-2"],
    });

    patchConversationInCache(queryClient, CONV_ID, {
      execution_status: ExecutionStatus.PAUSED,
      sandbox_status: "PAUSED",
    });

    const cached = queryClient.getQueryData<{
      pages: Array<{ items: AppConversation[]; next_page_id: string | null }>;
      pageParams: Array<string | null>;
    }>(queryKey);
    expect(cached?.pages[0]).toEqual({
      ...firstPage,
      items: [
        {
          ...firstMatch,
          execution_status: ExecutionStatus.PAUSED,
          sandbox_status: "PAUSED",
        },
        other,
      ],
    });
    expect(cached?.pages[1]).toEqual({
      ...secondPage,
      items: [
        {
          ...secondMatch,
          execution_status: ExecutionStatus.PAUSED,
          sandbox_status: "PAUSED",
        },
      ],
    });
    expect(cached?.pages[0]?.items[1]).toBe(other);
    expect(cached?.pageParams).toEqual([null, "page-2"]);
  });

  it("leaves registered caches without data empty", () => {
    const queryClient = makeQueryClient();
    const conversationKey = ["user", "conversation", CONV_ID];
    const conversationsKey = ["user", "conversations"];
    queryClient.getQueryCache().build(queryClient, {
      queryKey: conversationKey,
    });
    queryClient.getQueryCache().build(queryClient, {
      queryKey: conversationsKey,
    });

    patchConversationInCache(queryClient, CONV_ID, {
      sandbox_status: "PAUSED",
    });

    expect(queryClient.getQueryState(conversationKey)?.data).toBeUndefined();
    expect(queryClient.getQueryState(conversationsKey)?.data).toBeUndefined();
  });

  it("updates execution status without disturbing other cached fields", async () => {
    const { updateConversationExecutionStatusInCache: updateExecutionStatus } =
      await loadFreshConversationMutationUtils();
    const queryClient = makeQueryClient();
    const queryKey = ["user", "conversation", CONV_ID];
    const conversation = makeConversation({
      execution_status: ExecutionStatus.RUNNING,
      sandbox_status: "RUNNING",
    });
    queryClient.setQueryData(queryKey, conversation);

    updateExecutionStatus(queryClient, CONV_ID, ExecutionStatus.PAUSED);

    expect(queryClient.getQueryData(queryKey)).toEqual({
      ...conversation,
      execution_status: ExecutionStatus.PAUSED,
    });
  });

  it("updates the LLM model without disturbing other cached fields", async () => {
    const { updateConversationLlmModelInCache: updateLlmModel } =
      await loadFreshConversationMutationUtils();
    const queryClient = makeQueryClient();
    const queryKey = ["user", "conversation", CONV_ID];
    const conversation = makeConversation({
      llm_model: "provider/old-model",
      sandbox_status: "RUNNING",
    });
    queryClient.setQueryData(queryKey, conversation);

    updateLlmModel(queryClient, CONV_ID, "provider/new-model");

    expect(queryClient.getQueryData(queryKey)).toEqual({
      ...conversation,
      llm_model: "provider/new-model",
    });
  });

  it("invalidates every conversation-dependent query family", () => {
    const queryClient = makeQueryClient();
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");

    expect(invalidateConversationQueries(queryClient, CONV_ID)).toBeUndefined();

    expect(invalidateQueries.mock.calls).toEqual([
      [{ queryKey: ["user", "conversation", CONV_ID] }],
      [{ queryKey: ["user", "conversations"] }],
      [{ queryKey: ["v1-batch-get-app-conversations"] }],
      [{ queryKey: ["unified", "vscode_url"] }],
    ]);
  });
});
