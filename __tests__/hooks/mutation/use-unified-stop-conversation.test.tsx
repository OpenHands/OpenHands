import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUnifiedPauseConversation } from "#/hooks/mutation/use-unified-stop-conversation";
import { ExecutionStatus } from "#/types/agent-server/core";

const {
  pauseConversationMock,
  patchConversationInCacheMock,
  navigateMock,
  toastDismissMock,
  toastLoadingMock,
  toastSuccessMock,
} = vi.hoisted(() => ({
  pauseConversationMock: vi.fn(),
  patchConversationInCacheMock: vi.fn(),
  navigateMock: vi.fn(),
  toastDismissMock: vi.fn(),
  toastLoadingMock: vi.fn(() => "toast-id"),
  toastSuccessMock: vi.fn(),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  pauseConversation: (...args: unknown[]) => pauseConversationMock(...args),
  patchConversationInCache: (...args: unknown[]) =>
    patchConversationInCacheMock(...args),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    conversationId: "conv-1",
    navigate: navigateMock,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("react-hot-toast", () => ({
  default: {
    dismiss: toastDismissMock,
    loading: toastLoadingMock,
    success: toastSuccessMock,
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  TOAST_OPTIONS: {},
  displayErrorToast: vi.fn(),
}));

describe("useUnifiedPauseConversation", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    pauseConversationMock.mockReset().mockResolvedValue({ success: true });
    patchConversationInCacheMock.mockReset();
    navigateMock.mockReset();
    toastDismissMock.mockReset();
    toastLoadingMock.mockClear();
    toastSuccessMock.mockReset();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("keeps the active conversation open after stopping it", async () => {
    const { result } = renderHook(() => useUnifiedPauseConversation(), {
      wrapper,
    });

    await result.current.mutateAsync({ conversationId: "conv-1" });

    await waitFor(() => {
      expect(patchConversationInCacheMock).toHaveBeenCalledWith(
        queryClient,
        "conv-1",
        {
          execution_status: ExecutionStatus.PAUSED,
          sandbox_status: "PAUSED",
        },
      );
    });
    expect(navigateMock).not.toHaveBeenCalled();
  });
});
