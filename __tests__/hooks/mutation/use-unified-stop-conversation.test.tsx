import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError } from "axios";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUnifiedPauseConversation } from "#/hooks/mutation/use-unified-stop-conversation";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core";
import { TOAST_OPTIONS } from "#/utils/custom-toast-handlers";

const {
  displayErrorToastMock,
  navigateMock,
  navigationState,
  patchConversationInCacheMock,
  pauseConversationMock,
  toastDismissMock,
  toastLoadingMock,
  toastSuccessMock,
  useTranslationMock,
} = vi.hoisted(() => ({
  displayErrorToastMock: vi.fn(),
  navigateMock: vi.fn(),
  navigationState: { conversationId: "conversation-1" as string | null },
  patchConversationInCacheMock: vi.fn(),
  pauseConversationMock: vi.fn(),
  toastDismissMock: vi.fn(),
  toastLoadingMock: vi.fn(),
  toastSuccessMock: vi.fn(),
  useTranslationMock: vi.fn((namespace: string) => ({
    t: (key: string) => `translated:${key}`,
    namespace,
  })),
}));

vi.mock("react-hot-toast", () => ({
  default: {
    dismiss: (...args: unknown[]) => toastDismissMock(...args),
    loading: (...args: unknown[]) => toastLoadingMock(...args),
    success: (...args: unknown[]) => toastSuccessMock(...args),
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => useTranslationMock(namespace),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    conversationId: navigationState.conversationId,
    navigate: navigateMock,
  }),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  patchConversationInCache: (...args: unknown[]) =>
    patchConversationInCacheMock(...args),
  pauseConversation: (...args: unknown[]) => pauseConversationMock(...args),
}));

vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/custom-toast-handlers")>()),
  displayErrorToast: (...args: unknown[]) => displayErrorToastMock(...args),
}));

interface SetupOptions {
  currentConversationId?: string | null;
  previousConversations?: unknown;
}

function setup({
  currentConversationId = "conversation-1",
  previousConversations,
}: SetupOptions = {}) {
  navigationState.conversationId = currentConversationId;
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false },
    },
  });
  if (previousConversations !== undefined) {
    queryClient.setQueryData(["user", "conversations"], previousConversations);
  }
  const cancelQueries = vi.spyOn(queryClient, "cancelQueries");
  const setQueryData = vi.spyOn(queryClient, "setQueryData");
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  const hook = renderHook(() => useUnifiedPauseConversation(), { wrapper });

  return { ...hook, cancelQueries, queryClient, setQueryData };
}

beforeEach(() => {
  vi.clearAllMocks();
  navigationState.conversationId = "conversation-1";
  toastLoadingMock.mockReturnValue("stop-toast");
  pauseConversationMock.mockResolvedValue({ success: true });
});

describe("useUnifiedPauseConversation", () => {
  it("pauses the active conversation, patches both paused statuses, and navigates home", async () => {
    const previousConversations = {
      pages: [{ items: [{ id: "conversation-1" }] }],
    };
    const { cancelQueries, queryClient, result } = setup({
      previousConversations,
    });

    await act(async () => {
      await result.current.mutateAsync({ conversationId: "conversation-1" });
    });

    expect(useTranslationMock).toHaveBeenCalledWith("openhands");
    expect(
      queryClient.getMutationCache().getAll().at(-1)?.options.mutationKey,
    ).toEqual(["stop-conversation"]);
    expect(toastLoadingMock).toHaveBeenCalledWith(
      `translated:${I18nKey.TOAST$STOPPING_CONVERSATION}`,
      TOAST_OPTIONS,
    );
    expect(cancelQueries).toHaveBeenCalledWith({
      queryKey: ["user", "conversations"],
    });
    expect(cancelQueries.mock.invocationCallOrder[0]).toBeLessThan(
      pauseConversationMock.mock.invocationCallOrder[0]!,
    );
    expect(pauseConversationMock).toHaveBeenCalledWith("conversation-1");
    expect(toastDismissMock).toHaveBeenCalledWith("stop-toast");
    expect(toastSuccessMock).toHaveBeenCalledWith(
      `translated:${I18nKey.TOAST$CONVERSATION_STOPPED}`,
      TOAST_OPTIONS,
    );
    expect(patchConversationInCacheMock).toHaveBeenCalledWith(
      queryClient,
      "conversation-1",
      {
        execution_status: ExecutionStatus.PAUSED,
        sandbox_status: "PAUSED",
      },
    );
    expect(navigateMock).toHaveBeenCalledWith("/conversations");
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("does not dismiss a missing toast or navigate away from another conversation", async () => {
    toastLoadingMock.mockReturnValue("");
    const { queryClient, result } = setup({
      currentConversationId: "another-conversation",
    });

    await act(async () => {
      await result.current.mutateAsync({ conversationId: "conversation-1" });
    });

    expect(toastDismissMock).not.toHaveBeenCalled();
    expect(patchConversationInCacheMock).toHaveBeenCalledWith(
      queryClient,
      "conversation-1",
      {
        execution_status: ExecutionStatus.PAUSED,
        sandbox_status: "PAUSED",
      },
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("dismisses the loading toast and restores cached conversations on failure", async () => {
    const previousConversations = {
      pages: [{ items: [{ id: "conversation-1" }] }],
    };
    pauseConversationMock.mockRejectedValue(new Error("pause failed"));
    const { result, setQueryData } = setup({ previousConversations });

    await expect(
      act(async () =>
        result.current.mutateAsync({ conversationId: "conversation-1" }),
      ),
    ).rejects.toThrow("pause failed");

    expect(toastDismissMock).toHaveBeenCalledWith("stop-toast");
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      `translated:${I18nKey.TOAST$FAILED_TO_STOP_CONVERSATION}`,
    );
    expect(setQueryData).toHaveBeenCalledWith(
      ["user", "conversations"],
      previousConversations,
    );
    expect(toastSuccessMock).not.toHaveBeenCalled();
    expect(patchConversationInCacheMock).not.toHaveBeenCalled();
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("reports failure without restoring absent data or dismissing an absent toast", async () => {
    toastLoadingMock.mockReturnValue("");
    pauseConversationMock.mockRejectedValue(new Error("pause failed"));
    const { result, setQueryData } = setup();

    await expect(
      act(async () =>
        result.current.mutateAsync({ conversationId: "conversation-1" }),
      ),
    ).rejects.toThrow("pause failed");

    expect(toastDismissMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      `translated:${I18nKey.TOAST$FAILED_TO_STOP_CONVERSATION}`,
    );
    expect(setQueryData).not.toHaveBeenCalled();
  });

  it("reports an error safely when optimistic setup produced no context", async () => {
    const { queryClient, result, setQueryData } = setup();

    await act(async () => {
      await result.current.mutateAsync({ conversationId: "conversation-1" });
    });
    vi.clearAllMocks();

    const onError = queryClient.getMutationCache().getAll().at(-1)
      ?.options.onError;

    expect(onError).toBeTypeOf("function");
    expect(() =>
      onError?.(
        new AxiosError("optimistic setup failed"),
        { conversationId: "conversation-1" },
        undefined,
        {
          client: queryClient,
          meta: undefined,
          mutationKey: ["stop-conversation"],
        },
      ),
    ).not.toThrow();

    expect(displayErrorToastMock).toHaveBeenCalledWith(
      `translated:${I18nKey.TOAST$FAILED_TO_STOP_CONVERSATION}`,
    );
    expect(pauseConversationMock).not.toHaveBeenCalled();
    expect(toastDismissMock).not.toHaveBeenCalled();
    expect(setQueryData).not.toHaveBeenCalled();
  });
});
