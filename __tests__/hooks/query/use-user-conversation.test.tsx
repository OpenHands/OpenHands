import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// Mock the underlying service the hook depends on (not the hook itself).
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const CLOUD_CONVERSATION_ID = "conv-cloud";

function makeConversation(id: string): AppConversation {
  return {
    id,
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Test",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    execution_status: null,
    sandbox_status: null,
    conversation_url: "https://sandbox.example.com/api",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
  };
}

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockReset();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([makeConversation(CLOUD_CONVERSATION_ID)]);
  setRegisteredBackends([localBackend, cloudBackend]);
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useUserConversation — backend switch", () => {
  it("does not load the conversation id against a different backend after switching", async () => {
    // Arrange — the conversation is opened while the cloud backend is active.
    setActiveSelection({ backendId: cloudBackend.id });
    const { result } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);

    // Act — switch to the local backend without leaving the conversation.
    setActiveSelection({ backendId: localBackend.id });

    // Assert — the cloud id is foreign to the local backend, so the query is
    // disabled and never fetched against it (no second service call).
    await waitFor(() => expect(result.current.data).toBeUndefined());
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);
  });

  it("still loads the conversation when only the org changes on the same backend", async () => {
    // Arrange — opened under a cloud backend scoped to one org.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const { result } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);

    // Act — switch org within the same backend (e.g. the cloud personal-
    // workspace self-heal), which keeps the same agent-server schema.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-b" });

    // Assert — same backend, so the conversation is fetched again for the
    // new org rather than being disabled.
    await waitFor(() =>
      expect(
        AgentServerConversationService.batchGetAppConversations,
      ).toHaveBeenCalledTimes(2),
    );
  });
});
