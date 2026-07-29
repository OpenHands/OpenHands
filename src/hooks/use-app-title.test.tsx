import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useParams } from "react-router";
import { useUserConversation } from "./query/use-user-conversation";
import { useAppTitle } from "./use-app-title";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const renderAppTitleHook = () =>
  renderHook(() => useAppTitle(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

vi.mock("./query/use-user-conversation");
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useParams: vi.fn(),
  };
});

describe("useAppTitle", () => {
  const mockUseUserConversation = vi.mocked(useUserConversation);
  const mockUseParams = vi.mocked(useParams);

  beforeEach(() => {
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: null });
    mockUseParams.mockReturnValue({});
    useConversationStateStore.getState().reset();
  });

  it("returns the OSS app title outside conversations", async () => {
    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });

  it("returns the conversation title with the OSS app name", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "My Conversation" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("My Conversation | OpenHands"),
    );
  });

  it("returns the app name while conversation data is loading", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: undefined });

    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });

  it.each([
    [ExecutionStatus.RUNNING, "🟢"],
    [ExecutionStatus.FINISHED, "✅"],
    [ExecutionStatus.IDLE, "✅"],
    [ExecutionStatus.WAITING_FOR_CONFIRMATION, "✅"],
    [ExecutionStatus.PAUSED, "⚪"],
    [ExecutionStatus.ERROR, "🔴"],
    [ExecutionStatus.STUCK, "🔴"],
  ])(
    "prefixes the title with %s emoji for execution status %s",
    async (status, emoji) => {
      mockUseParams.mockReturnValue({ conversationId: "123" });
      mockUseUserConversation.mockReturnValue({
        // @ts-expect-error - only returning partial config for test
        data: { title: "My Conversation" },
      });
      useConversationStateStore.getState().setExecutionStatus(status);

      const { result } = renderAppTitleHook();

      await waitFor(() =>
        expect(result.current).toBe(`${emoji} My Conversation | OpenHands`),
      );
    },
  );

  it("falls back to the conversation's execution_status when the live store is empty", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: {
        title: "My Conversation",
        execution_status: ExecutionStatus.RUNNING,
      },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("🟢 My Conversation | OpenHands"),
    );
  });

  it("does not add an emoji when in a conversation but execution status is unknown", async () => {
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "My Conversation" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("My Conversation | OpenHands"),
    );
  });
});
