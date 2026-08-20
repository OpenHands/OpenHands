import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useUpdateConversationRepository } from "#/hooks/mutation/use-update-conversation-repository";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// Mock the AgentServerConversationService
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      updateConversationRepository: vi.fn(),
    },
  }),
);

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

// Mock toast handlers
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const createWrapper = (queryClient?: QueryClient) => {
  const client =
    queryClient ??
    new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );

  return { Wrapper, client };
};

describe("useUpdateConversationRepository", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call updateConversationRepository with correct parameters", async () => {
    const mockResponse = {
      id: "test-conversation-id",
      selected_repository: "owner/repo",
      selected_branch: "main",
      git_provider: "github",
    };

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockResolvedValue(mockResponse as any);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId: "test-conversation-id",
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(
      AgentServerConversationService.updateConversationRepository,
    ).toHaveBeenCalledWith(
      "test-conversation-id",
      "owner/repo",
      "main",
      "github",
    );
  });

  it("should handle repository removal (null values)", async () => {
    const mockResponse = {
      id: "test-conversation-id",
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    };

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockResolvedValue(mockResponse as any);

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId: "test-conversation-id",
      repository: null,
      branch: null,
      gitProvider: null,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(
      AgentServerConversationService.updateConversationRepository,
    ).toHaveBeenCalledWith("test-conversation-id", null, null, null);
  });

  it("should handle errors gracefully", async () => {
    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockRejectedValue(new Error("Failed to update repository"));

    const { Wrapper } = createWrapper();
    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId: "test-conversation-id",
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });

  it("optimistically updates the cached conversation under the prefix-extended key used by useUserConversation", async () => {
    const conversationId = "test-conversation-id";
    // Mirror the key shape from `useUserConversation`:
    //   ["user", "conversation", cid, backendId, orgId]
    const cacheKey = [
      "user",
      "conversation",
      conversationId,
      "default-local",
      null,
    ] as const;
    const baseConversation = {
      id: conversationId,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    } as unknown as AppConversation;

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockImplementation(async () => {
      // Block long enough to assert the optimistic update fires before
      // the mutation resolves.
      await new Promise((resolve) => setTimeout(resolve, 50));
      return {
        ...baseConversation,
        selected_repository: "owner/repo",
        selected_branch: "main",
        git_provider: "github",
      } as unknown as AppConversation;
    });

    const { Wrapper, client } = createWrapper();
    client.setQueryData(cacheKey, baseConversation);

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId,
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      const cached = client.getQueryData<AppConversation>(cacheKey);
      expect(cached?.selected_repository).toBe("owner/repo");
      expect(cached?.selected_branch).toBe("main");
      expect(cached?.git_provider).toBe("github");
    });
  });

  it("optimistically updates every prefix-matching cache entry (multiple backends/orgs)", async () => {
    const conversationId = "test-conversation-id";
    const cacheKey1 = [
      "user",
      "conversation",
      conversationId,
      "backend-1",
      "org-1",
    ] as const;
    const cacheKey2 = [
      "user",
      "conversation",
      conversationId,
      "backend-2",
      "org-2",
    ] as const;
    const baseConversation = {
      id: conversationId,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    } as unknown as AppConversation;

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
      return {
        ...baseConversation,
        selected_repository: "owner/repo",
        selected_branch: "main",
        git_provider: "github",
      } as unknown as AppConversation;
    });

    const { Wrapper, client } = createWrapper();
    client.setQueryData(cacheKey1, baseConversation);
    client.setQueryData(cacheKey2, baseConversation);

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId,
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      const cached1 = client.getQueryData<AppConversation>(cacheKey1);
      const cached2 = client.getQueryData<AppConversation>(cacheKey2);
      expect(cached1?.selected_repository).toBe("owner/repo");
      expect(cached1?.selected_branch).toBe("main");
      expect(cached1?.git_provider).toBe("github");
      expect(cached2?.selected_repository).toBe("owner/repo");
      expect(cached2?.selected_branch).toBe("main");
      expect(cached2?.git_provider).toBe("github");
    });
  });

  it("rolls back every prefix-matching cache entry when the mutation rejects", async () => {
    const conversationId = "test-conversation-id";
    const cacheKey1 = [
      "user",
      "conversation",
      conversationId,
      "backend-1",
      "org-1",
    ] as const;
    const cacheKey2 = [
      "user",
      "conversation",
      conversationId,
      "backend-2",
      "org-2",
    ] as const;
    const baseConversation = {
      id: conversationId,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    } as unknown as AppConversation;

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockRejectedValue(new Error("nope"));

    const { Wrapper, client } = createWrapper();
    client.setQueryData(cacheKey1, baseConversation);
    client.setQueryData(cacheKey2, baseConversation);

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId,
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    const cached1 = client.getQueryData<AppConversation>(cacheKey1);
    const cached2 = client.getQueryData<AppConversation>(cacheKey2);
    expect(cached1?.selected_repository).toBeNull();
    expect(cached1?.selected_branch).toBeNull();
    expect(cached1?.git_provider).toBeNull();
    expect(cached2?.selected_repository).toBeNull();
    expect(cached2?.selected_branch).toBeNull();
    expect(cached2?.git_provider).toBeNull();
  });

  it("invalidates user-conversation, user-conversations, and local-git-info on settle", async () => {
    const conversationId = "test-conversation-id";

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockResolvedValue({
      id: conversationId,
      selected_repository: "owner/repo",
      selected_branch: "main",
      git_provider: "github",
    } as unknown as AppConversation);

    const { Wrapper, client } = createWrapper();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    await result.current.mutateAsync({
      conversationId,
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", conversationId],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversations"],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["local-git-info", conversationId],
      });
    });
  });

  it("still invalidates all three query keys on settle when the mutation rejects", async () => {
    const conversationId = "test-conversation-id";

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockRejectedValue(new Error("boom"));

    const { Wrapper, client } = createWrapper();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    await expect(
      result.current.mutateAsync({
        conversationId,
        repository: "owner/repo",
        branch: "main",
        gitProvider: "github",
      }),
    ).rejects.toThrow("boom");

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", conversationId],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversations"],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["local-git-info", conversationId],
      });
    });
  });

  it("rolls back the prefix-keyed cache entry when the mutation rejects", async () => {
    const conversationId = "test-conversation-id";
    const cacheKey = [
      "user",
      "conversation",
      conversationId,
      "default-local",
      null,
    ] as const;
    const baseConversation = {
      id: conversationId,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
    } as unknown as AppConversation;

    vi.mocked(
      AgentServerConversationService.updateConversationRepository,
    ).mockRejectedValue(new Error("nope"));

    const { Wrapper, client } = createWrapper();
    client.setQueryData(cacheKey, baseConversation);

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: Wrapper,
    });

    result.current.mutate({
      conversationId,
      repository: "owner/repo",
      branch: "main",
      gitProvider: "github",
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    const cached = client.getQueryData<AppConversation>(cacheKey);
    expect(cached?.selected_repository).toBeNull();
    expect(cached?.selected_branch).toBeNull();
    expect(cached?.git_provider).toBeNull();
  });
});
