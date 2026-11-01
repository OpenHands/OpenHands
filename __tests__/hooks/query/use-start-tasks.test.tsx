import { HttpError } from "@openhands/typescript-client";
import React from "react";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useStartTasks } from "#/hooks/query/use-start-tasks";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { getStartTask: vi.fn() } }),
);
let client: QueryClient;
const working = {
  id: "123",
  status: "WORKING",
  request: {},
  app_conversation_id: null,
};
function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
}
beforeEach(() => {
  vi.clearAllMocks();
  client = new QueryClient();
  setRegisteredBackends([
    {
      id: "cloud",
      name: "Cloud",
      host: "https://example.test",
      apiKey: "",
      kind: "cloud",
    },
  ]);
  setActiveSelection({ backendId: "cloud", orgId: "org-1" });
  useConversationLaunchStore.setState({ launches: {} });
  for (const orgId of ["org-1", "org-2"])
    useConversationLaunchStore.getState().begin({
      backendId: "cloud",
      orgId,
      message: "Hello",
      files: [],
      targetId: orgId === "org-1" ? "task-123" : "task-other",
    });
  vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
    working as never,
  );
});
afterEach(() => {
  client.clear();
  setRegisteredBackends([]);
  vi.useRealTimers();
});

it("shows only this organization's pending starts and refreshes conversations when ready", async () => {
  const list = vi
    .fn()
    .mockResolvedValueOnce([])
    .mockResolvedValue(["new-conversation"]);
  const { result } = renderHook(
    () => ({
      tasks: useStartTasks().data,
      conversations: useQuery({
        queryKey: ["user", "conversations"],
        queryFn: list,
      }).data,
    }),
    { wrapper },
  );
  await waitFor(() => expect(result.current.tasks).toHaveLength(1));
  expect(
    AgentServerConversationService.getStartTask,
  ).toHaveBeenCalledExactlyOnceWith("123");
  act(() =>
    client.setQueryData(["start-task", "123"], {
      ...working,
      status: "READY",
      app_conversation_id: "new-conversation",
    }),
  );
  await waitFor(() =>
    expect(result.current.conversations).toEqual(["new-conversation"]),
  );
  expect(result.current.tasks).toEqual([]);
});

it("stops polling a completed background start", async () => {
  vi.useFakeTimers();
  vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue({
    ...working,
    status: "READY",
    app_conversation_id: "new-conversation",
  } as never);
  renderHook(() => useStartTasks(), { wrapper });
  await act(() => vi.advanceTimersByTimeAsync(30000));
  expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(1);
});

it("keeps an acknowledged start findable when its first status read fails", async () => {
  vi.mocked(AgentServerConversationService.getStartTask).mockRejectedValue(
    new HttpError(403, "Forbidden", { detail: "Cannot read status" }),
  );
  const { result } = renderHook(() => useStartTasks(), { wrapper });
  await waitFor(() => expect(result.current.failedTaskIds).toEqual(["123"]));
  expect(result.current.unresolvedTaskIds).toEqual(["123"]);
});
