import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import { useConversationHistory } from "#/hooks/query/use-conversation-history";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { callCloudProxy } from "#/api/cloud/proxy";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import EventService from "#/api/event-service/event-service.api";
import { createAgentServerQueryClient } from "#/query-client-config";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: vi.fn() }));
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

let client: ReturnType<typeof createAgentServerQueryClient>;
function wrapper({ children }: React.PropsWithChildren) {
  return (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([
    {
      id: "cloud",
      name: "Cloud",
      kind: "cloud",
      host: "https://app.example.com",
      apiKey: "test-key",
    },
  ]);
  setActiveSelection({ backendId: "cloud", orgId: "org-test" });
  client = createAgentServerQueryClient();
  client.setDefaultOptions({ queries: { retryDelay: 0 } });
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([
    {
      id: "conversation",
      conversation_url: "https://runtime.example.com",
      session_api_key: "runtime-key",
      sandbox_status: "RUNNING",
    } as Awaited<
      ReturnType<typeof AgentServerConversationService.batchGetAppConversations>
    >[number],
  ]);
  vi.spyOn(EventService, "getEventCount").mockResolvedValue(120);
});

afterEach(() => {
  client.clear();
  vi.restoreAllMocks();
  vi.resetAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("Cloud initial history recovery", () => {
  it("retries a transient failure and keeps the recovered page and pagination without a toast", async () => {
    const oldest = { id: "older", timestamp: "2026-01-01T00:00:00Z" };
    const newest = { id: "newer", timestamp: "2026-01-01T00:01:00Z" };
    vi.mocked(callCloudProxy)
      .mockRejectedValueOnce(new HttpError(503, "Temporarily unavailable"))
      .mockResolvedValueOnce({
        items: [newest, oldest],
        next_page_id: "older-page",
      });

    const { result } = renderHook(
      () => useConversationHistory("conversation"),
      { wrapper },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(callCloudProxy).toHaveBeenCalledTimes(2);
    expect(result.current.data).toMatchObject({
      events: [oldest, newest],
      hasMore: true,
      nextPageId: "older-page",
      afterSeq: 69,
    });
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("settles a persistent failure as an error after one retry so socket recovery can proceed quietly", async () => {
    const failure = new HttpError(503, "Unavailable");
    vi.mocked(callCloudProxy).mockRejectedValue(failure);

    const { result } = renderHook(
      () => useConversationHistory("conversation"),
      { wrapper },
    );
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(callCloudProxy).toHaveBeenCalledTimes(2);
    expect(result.current.isPending).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.error).toBe(failure);
    expect(displayErrorToast).not.toHaveBeenCalled();
  });
});
