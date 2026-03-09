import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useUpdateConversationRepository } from "#/hooks/mutation/use-update-conversation-repository";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

// Mock the V1ConversationService
vi.mock("#/api/conversation-service/v1-conversation-service.api", () => ({
  default: {
    updateConversationRepository: vi.fn(),
  },
}));

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

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
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

    vi.mocked(V1ConversationService.updateConversationRepository).mockResolvedValue(
      mockResponse as any,
    );

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: createWrapper(),
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

    expect(V1ConversationService.updateConversationRepository).toHaveBeenCalledWith(
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

    vi.mocked(V1ConversationService.updateConversationRepository).mockResolvedValue(
      mockResponse as any,
    );

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: createWrapper(),
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

    expect(V1ConversationService.updateConversationRepository).toHaveBeenCalledWith(
      "test-conversation-id",
      null,
      null,
      null,
    );
  });

  it("should handle errors gracefully", async () => {
    vi.mocked(V1ConversationService.updateConversationRepository).mockRejectedValue(
      new Error("Failed to update repository"),
    );

    const { result } = renderHook(() => useUpdateConversationRepository(), {
      wrapper: createWrapper(),
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
});
