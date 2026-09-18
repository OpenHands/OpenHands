import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import SkillsService from "#/api/skills-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import type { SkillInfo } from "#/types/settings";

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
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

const CONVERSATION_ID = "conv-1";
const WORKSPACE = "/workspace/project/demo";

function makeConversation(
  overrides: Partial<AppConversation> = {},
): AppConversation {
  return {
    id: CONVERSATION_ID,
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
    sandbox_status: "RUNNING",
    conversation_url: "https://sandbox.example.com/api",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    selected_workspace: WORKSPACE,
    ...overrides,
  };
}

function makeSkill(name: string): SkillInfo {
  return {
    name,
    type: "agentskills",
    source: null,
    content: `# ${name}`,
    triggers: [],
  };
}

function makeWrapper(conversationId: string | null) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              currentPath: "/",
              conversationId,
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            {children}
          </NavigationProvider>
        </ActiveBackendProvider>
      </QueryClientProvider>
    );
  }
  return { wrapper: Wrapper, queryClient };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend, cloudBackend]);
  // Mock the services the hook depends on, not the hooks themselves.
  vi.spyOn(
    AgentServerConversationService,
    "batchGetAppConversations",
  ).mockResolvedValue([makeConversation()]);
  vi.spyOn(SkillsService, "getConversationSkills").mockResolvedValue([
    makeSkill("release-notes"),
  ]);
  vi.spyOn(SkillsService, "getSkills").mockResolvedValue([
    makeSkill("catalog-only"),
  ]);
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useConversationSkills", () => {
  it("lists the conversation's own skills on a cloud backend once its sandbox runs", async () => {
    // Arrange
    setActiveSelection({ backendId: cloudBackend.id });
    const { wrapper } = makeWrapper(CONVERSATION_ID);

    // Act
    const { result } = renderHook(() => useConversationSkills(), { wrapper });

    // Assert
    await waitFor(() =>
      expect(result.current.data).toEqual([makeSkill("release-notes")]),
    );
    expect(SkillsService.getConversationSkills).toHaveBeenCalledWith(
      CONVERSATION_ID,
    );
    expect(SkillsService.getSkills).not.toHaveBeenCalled();
  });

  it("does not request skills for a cloud conversation whose sandbox is not running yet", async () => {
    // Arrange
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([makeConversation({ sandbox_status: "STARTING" })]);
    const { wrapper, queryClient } = makeWrapper(CONVERSATION_ID);

    // Act
    const { result } = renderHook(() => useConversationSkills(), { wrapper });

    // Assert — even once the conversation itself has loaded.
    await waitFor(() => {
      const [entry] = queryClient.getQueriesData<AppConversation | null>({
        queryKey: ["user", "conversation"],
      });
      expect(entry?.[1]?.sandbox_status).toBe("STARTING");
    });
    expect(result.current.fetchStatus).toBe("idle");
    expect(result.current.data).toBeUndefined();
    expect(SkillsService.getConversationSkills).not.toHaveBeenCalled();
    expect(SkillsService.getSkills).not.toHaveBeenCalled();
  });

  it("keeps the workspace-scoped catalog on a local backend", async () => {
    // Arrange
    setActiveSelection({ backendId: localBackend.id });
    const { wrapper } = makeWrapper(CONVERSATION_ID);

    // Act
    const { result } = renderHook(() => useConversationSkills(), { wrapper });

    // Assert
    await waitFor(() =>
      expect(SkillsService.getSkills).toHaveBeenCalledWith(WORKSPACE),
    );
    await waitFor(() =>
      expect(result.current.data).toEqual([makeSkill("catalog-only")]),
    );
    expect(SkillsService.getConversationSkills).not.toHaveBeenCalled();
  });

  it("falls back to the global catalog on a cloud backend without a conversation route", async () => {
    // Arrange
    setActiveSelection({ backendId: cloudBackend.id });
    const { wrapper } = makeWrapper(null);

    // Act
    const { result } = renderHook(() => useConversationSkills(), { wrapper });

    // Assert
    await waitFor(() =>
      expect(result.current.data).toEqual([makeSkill("catalog-only")]),
    );
    expect(SkillsService.getSkills).toHaveBeenCalledWith(undefined);
    expect(SkillsService.getConversationSkills).not.toHaveBeenCalled();
  });
});
