import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { CONVERSATION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { NavigationProvider } from "#/context/navigation-context";
import type { WebSocketConnectionState } from "#/contexts/conversation-websocket-context";
import { I18nKey } from "#/i18n/declaration";
import { useSandboxRestartWarning } from "#/hooks/use-sandbox-restart-warning";

const mocks = vi.hoisted(() => ({
  connectionState: "OPEN" as WebSocketConnectionState,
  displayErrorToast: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { reason?: string }) =>
      options?.reason ? `${key}:${options.reason}` : key,
  }),
}));

vi.mock("#/contexts/conversation-websocket-context", () => ({
  useConversationWebSocket: () => ({
    connectionState: mocks.connectionState,
    mainConnectionState: mocks.connectionState,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mocks.displayErrorToast,
}));

const restartToast = (reason: string) =>
  `${I18nKey.CONVERSATION$SANDBOX_RESTARTED}:${reason}`;

const conversation = (
  restartCount: number | null | undefined,
  restartReasons: string[] = [],
  id = "conversation-1",
) =>
  ({
    id,
    title: "Test",
    created_at: new Date(0).toISOString(),
    sandbox_status: "RUNNING",
    conversation_url: "http://runtime.test",
    sandbox_restart_count: restartCount,
    sandbox_restart_reasons: restartReasons,
  }) as AppConversation;

describe("useSandboxRestartWarning", () => {
  let queryClient: QueryClient;
  let conversationId: string;
  let serverConversations: Record<string, AppConversation>;

  const conversationQueryKey = (id: string) =>
    CONVERSATION_QUERY_KEYS.detail(
      id,
      getActiveBackend().backend.id,
      getActiveBackend().orgId,
    );

  const renderWarning = () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <NavigationProvider
          value={{
            currentPath: `/conversations/${conversationId}`,
            conversationId,
            isNavigating: false,
            navigate: vi.fn(),
          }}
        >
          {children}
        </NavigationProvider>
      </QueryClientProvider>
    );
    return renderHook(() => useSandboxRestartWarning(), { wrapper });
  };

  const poll = async () => {
    await queryClient.refetchQueries({ type: "active" });
    // Query observers are notified on a macrotask, after the refetch resolves.
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    conversationId = "conversation-1";
    serverConversations = {};
    mocks.connectionState = "OPEN";
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockImplementation(async (ids) =>
      ids.map((id) => serverConversations[id] ?? null),
    );
  });

  const renderAfterFirstRead = async (first: AppConversation) => {
    serverConversations[first.id] = first;
    const view = renderWarning();
    await waitFor(() =>
      expect(queryClient.getQueryData(conversationQueryKey(first.id))).toEqual(
        first,
      ),
    );
    return view;
  };

  describe("restart detection", () => {
    it("does not warn about restarts that happened before the first read", async () => {
      await renderAfterFirstRead(conversation(3, ["OOMKilled"]));

      expect(mocks.displayErrorToast).not.toHaveBeenCalled();
    });

    it("warns with the reason when the restart count rises", async () => {
      await renderAfterFirstRead(conversation(0));

      serverConversations["conversation-1"] = conversation(1, ["OOMKilled"]);
      await poll();

      expect(mocks.displayErrorToast).toHaveBeenCalledTimes(1);
      expect(mocks.displayErrorToast).toHaveBeenCalledWith(
        restartToast("OOMKilled"),
      );
    });

    it("warns once per restart, not once per poll", async () => {
      await renderAfterFirstRead(conversation(0));

      serverConversations["conversation-1"] = conversation(1, ["Error"]);
      await poll();
      await poll();
      expect(mocks.displayErrorToast).toHaveBeenCalledTimes(1);

      serverConversations["conversation-1"] = conversation(2, ["OOMKilled"]);
      await poll();
      expect(mocks.displayErrorToast).toHaveBeenCalledTimes(2);
    });

    it("falls back to an unknown reason when none is reported", async () => {
      await renderAfterFirstRead(conversation(0));

      serverConversations["conversation-1"] = conversation(1, []);
      await poll();

      expect(mocks.displayErrorToast).toHaveBeenCalledWith(
        restartToast("Unknown"),
      );
    });

    it("starts a new baseline when the count drops because the pod was replaced", async () => {
      await renderAfterFirstRead(conversation(3));

      serverConversations["conversation-1"] = conversation(0);
      await poll();
      expect(mocks.displayErrorToast).not.toHaveBeenCalled();

      serverConversations["conversation-1"] = conversation(1, ["OOMKilled"]);
      await poll();
      expect(mocks.displayErrorToast).toHaveBeenCalledTimes(1);
    });

    it("keeps the baseline across an unknown count", async () => {
      await renderAfterFirstRead(conversation(2));

      serverConversations["conversation-1"] = conversation(null);
      await poll();
      serverConversations["conversation-1"] = conversation(undefined);
      await poll();
      expect(mocks.displayErrorToast).not.toHaveBeenCalled();

      serverConversations["conversation-1"] = conversation(3, ["OOMKilled"]);
      await poll();
      expect(mocks.displayErrorToast).toHaveBeenCalledTimes(1);
    });

    it("does not compare counts across different conversations", async () => {
      serverConversations["conversation-2"] = conversation(
        4,
        ["OOMKilled"],
        "conversation-2",
      );
      const { rerender } = await renderAfterFirstRead(conversation(0));

      conversationId = "conversation-2";
      rerender();
      await waitFor(() =>
        expect(
          AgentServerConversationService.batchGetAppConversations,
        ).toHaveBeenCalledWith(["conversation-2"]),
      );
      await poll();

      expect(mocks.displayErrorToast).not.toHaveBeenCalled();
    });
  });

  describe("WebSocket drops", () => {
    it("refetches the conversation when an open connection closes", async () => {
      const { rerender } = await renderAfterFirstRead(conversation(0));
      const service = vi.mocked(
        AgentServerConversationService.batchGetAppConversations,
      );
      service.mockClear();

      serverConversations["conversation-1"] = conversation(1, ["OOMKilled"]);
      mocks.connectionState = "CLOSED";
      rerender();

      await waitFor(() => expect(service).toHaveBeenCalledTimes(1));
      await waitFor(() =>
        expect(mocks.displayErrorToast).toHaveBeenCalledWith(
          restartToast("OOMKilled"),
        ),
      );
    });

    it("does not refetch when a connection that never opened is closed", async () => {
      mocks.connectionState = "CONNECTING";
      const { rerender } = await renderAfterFirstRead(conversation(0));
      const service = vi.mocked(
        AgentServerConversationService.batchGetAppConversations,
      );
      service.mockClear();

      mocks.connectionState = "CLOSED";
      rerender();

      await act(async () => {});
      expect(service).not.toHaveBeenCalled();
    });
  });
});
