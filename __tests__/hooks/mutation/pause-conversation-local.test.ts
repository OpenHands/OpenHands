import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { pauseConversation } from "#/hooks/mutation/conversation-mutation-utils";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const { interruptConversationMock } = vi.hoisted(() => ({
  interruptConversationMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(function ConversationClientMock() {
    return { interruptConversation: interruptConversationMock };
  }),
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:18000",
  apiKey: "test-key",
  kind: "local",
};

const buildConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: "conv-local-1",
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2026-04-16T00:00:00Z",
  updated_at: "2026-04-16T00:00:00Z",
  execution_status: ExecutionStatus.RUNNING,
  conversation_url: "http://localhost:18000",
  session_api_key: "sess-key",
  sandbox_id: null,
  sub_conversation_ids: [],
  ...overrides,
});

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  interruptConversationMock.mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.restoreAllMocks();
});

describe("pauseConversation local branch", () => {
  it("calls interruptConversation on the local agent server", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([buildConversation()]);
    interruptConversationMock.mockResolvedValue({ success: true });

    await pauseConversation("conv-local-1");

    expect(interruptConversationMock).toHaveBeenCalledOnce();
    expect(interruptConversationMock).toHaveBeenCalledWith("conv-local-1");
  });

  it("propagates errors from interruptConversation", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([buildConversation()]);
    interruptConversationMock.mockRejectedValue(new Error("interrupt failed"));

    await expect(pauseConversation("conv-local-1")).rejects.toThrow(
      "interrupt failed",
    );
  });
});
