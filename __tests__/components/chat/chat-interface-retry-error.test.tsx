import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatInterface } from "#/components/features/chat/chat-interface";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

const mockResumeConversation = vi.fn();
let mockIsPending = false;

vi.mock("#/hooks/mutation/use-unified-start-conversation", () => ({
  useUnifiedResumeConversation: () => ({
    mutate: mockResumeConversation,
    isPending: mockIsPending,
  }),
}));

const mockToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => mockToast(...args),
}));

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: vi.fn() }),
}));
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: {} }),
}));
vi.mock("#/hooks/mutation/use-unified-upload-files", () => ({
  useUnifiedUploadFiles: () => ({
    mutateAsync: vi.fn(),
    isLoading: false,
  }),
}));
vi.mock("#/hooks/use-llm-configured", () => ({
  useLlmConfigured: () => ({ isConfigured: true, isLoading: false }),
}));
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conv-id" }),
  useOptionalConversationId: () => ({ conversationId: "test-conv-id" }),
}));
vi.mock("#/hooks/use-agent-state", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/hooks/use-agent-state")>();
  return {
    ...actual,
    useAgentState: vi.fn(),
    usePlanningAgentState: () => ({ isPlanningAgentRunning: false }),
  };
});
vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: [] }),
}));
vi.mock("#/hooks/use-load-older-events", () => ({
  useLoadOlderEvents: () => ({
    isLoading: false,
    hasMore: false,
    loadOlder: vi.fn(),
    scrollTargetIndex: null,
  }),
}));
vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({ isTask: false }),
}));

describe("ChatInterface - Retry transient LLM provider API errors", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    mockResumeConversation.mockReset();
    mockToast.mockReset();
    mockIsPending = false;
    useErrorMessageStore.getState().removeErrorMessage();
    (useAgentState as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      curAgentState: AgentState.ERROR,
    });
  });

  afterEach(() => {
    queryClient.clear();
    useErrorMessageStore.getState().removeErrorMessage();
  });

  function renderInterface() {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/test-conv-id"]}>
          <Routes>
            <Route path=":conversationId" element={<ChatInterface />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  }

  it("shows retry button when error has retryable: true and triggers resumeConversation", async () => {
    const user = userEvent.setup();
    useErrorMessageStore
      .getState()
      .setErrorMessage(
        "litellm.InternalServerError - Flex processing is temporarily unavailable",
        "conversation",
        null,
        {
          kind: "transient",
          retryable: true,
          user_action: "retry",
        },
      );

    renderInterface();

    const retryButton = screen.getByTestId("error-message-banner-retry");
    expect(retryButton).toBeVisible();

    await user.click(retryButton);
    expect(mockResumeConversation).toHaveBeenCalledTimes(1);
    expect(mockResumeConversation).toHaveBeenCalledWith(
      { conversationId: "test-conv-id" },
      expect.objectContaining({ onError: expect.any(Function) }),
    );
  });

  it("omits retry button when error is non-retryable auth error and shows re-auth instead", () => {
    useErrorMessageStore
      .getState()
      .setErrorMessage("Invalid API key", "conversation", "ACPAuthRequired", {
        kind: "auth",
        retryable: false,
        user_action: "settings",
      });

    renderInterface();

    expect(
      screen.queryByTestId("error-message-banner-retry"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("error-message-banner-reauth"),
    ).toBeInTheDocument();
  });

  it("omits retry button when error is non-retryable quota error", () => {
    useErrorMessageStore
      .getState()
      .setErrorMessage("Usage quota exceeded", "conversation", null, {
        kind: "quota",
        retryable: false,
        user_action: "none",
      });

    renderInterface();

    expect(
      screen.queryByTestId("error-message-banner-retry"),
    ).not.toBeInTheDocument();
  });

  it("hides retry button when agent is currently running", () => {
    (useAgentState as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
    useErrorMessageStore
      .getState()
      .setErrorMessage(
        "Flex processing is temporarily unavailable",
        "conversation",
        null,
        {
          kind: "transient",
          retryable: true,
          user_action: "retry",
        },
      );

    renderInterface();

    expect(
      screen.queryByTestId("error-message-banner-retry"),
    ).not.toBeInTheDocument();
  });

  it("surfaces 409 error on retry without corrupting state", async () => {
    const user = userEvent.setup();
    mockResumeConversation.mockImplementation((_, { onError }) => {
      onError(new Error("409 Conflict: Run already in progress"));
    });

    useErrorMessageStore
      .getState()
      .setErrorMessage("Temporary gateway timeout", "conversation", null, {
        kind: "transient",
        retryable: true,
        user_action: "retry",
      });

    renderInterface();

    const retryButton = screen.getByTestId("error-message-banner-retry");
    await user.click(retryButton);

    expect(mockToast).toHaveBeenCalledWith(
      "409 Conflict: Run already in progress",
    );
    expect(useErrorMessageStore.getState().errorMessage).toBe(
      "Temporary gateway timeout",
    );
  });
});
