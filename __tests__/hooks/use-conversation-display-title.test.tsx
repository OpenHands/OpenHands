import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { NavigationProvider } from "#/context/navigation-context";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { useConversationDisplayTitle } from "#/hooks/use-conversation-display-title";
import { CONVERSATION_QUERY_KEYS } from "#/hooks/query/query-keys";

let client: QueryClient;
let launchId: string;
let routeId: string;
const prompt = "Explain the startup flow";
const id = "12345-real";
const listKey = CONVERSATION_QUERY_KEYS.paginated(20, "cloud", "org-1");
const detailKey = CONVERSATION_QUERY_KEYS.detail(id, "cloud", "org-1");
const metadata = (title: string, updated_at = "2026-11-01T10:00:00Z") => ({
  id,
  title,
  updated_at,
});
function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <NavigationProvider
          value={{
            conversationId: routeId,
            currentPath: `/conversations/${routeId}`,
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
beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
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
  launchId = useConversationLaunchStore
    .getState()
    .begin({ backendId: "cloud", orgId: "org-1", files: [], message: prompt });
  routeId = launchId;
});
afterEach(() => {
  client.clear();
  useConversationLaunchStore.setState({ launches: {} });
  setRegisteredBackends([]);
});

it("keeps both titles through launch, task and saved metadata, then adopts generated and renamed titles together", async () => {
  const { result, rerender } = renderHook(
    () => {
      const header = useConversationDisplayTitle();
      const sidebar = useConversationDisplayTitle();
      return [header(routeId), sidebar(routeId)];
    },
    { wrapper },
  );
  expect(result.current).toEqual([prompt, prompt]);
  act(() =>
    useConversationLaunchStore
      .getState()
      .update(launchId, { targetId: "task-start" }),
  );
  routeId = "task-start";
  rerender();
  expect(result.current).toEqual([prompt, prompt]);
  act(() =>
    useConversationLaunchStore
      .getState()
      .update(launchId, { conversationId: id }),
  );
  routeId = id;
  rerender();
  expect(result.current).toEqual([prompt, prompt]);
  act(() => {
    client.setQueryData(detailKey, metadata("Conversation 12345"));
    client.setQueryData(listKey, {
      pages: [{ items: [metadata("Conversation 12345")] }],
      pageParams: [undefined],
    });
  });
  await waitFor(() =>
    expect(client.getQueryState(detailKey)?.data).toBeDefined(),
  );
  expect(result.current).toEqual([prompt, prompt]);
  act(() =>
    client.setQueryData(
      detailKey,
      metadata("Startup explained", "2026-11-01T10:01:00Z"),
    ),
  );
  await waitFor(() =>
    expect(result.current).toEqual(["Startup explained", "Startup explained"]),
  );
  act(() =>
    client.setQueryData(listKey, {
      pages: [{ items: [metadata("My custom title", "2026-11-01T10:02:00Z")] }],
      pageParams: [undefined],
    }),
  );
  await waitFor(() =>
    expect(result.current).toEqual(["My custom title", "My custom title"]),
  );
  expect(client.isFetching()).toBe(0);
});

it("keeps a generated title in the sidebar after leaving the conversation and isolates organizations", () => {
  useConversationLaunchStore
    .getState()
    .update(launchId, { conversationId: id });
  client.setQueryData(detailKey, metadata("Startup explained"));
  routeId = "another-chat";
  const { result } = renderHook(() => useConversationDisplayTitle()(id), {
    wrapper,
  });
  expect(result.current).toBe("Startup explained");
  act(() => setActiveSelection({ backendId: "cloud", orgId: "org-2" }));
  expect(result.current).not.toBe("Startup explained");
  expect(result.current).not.toBe(prompt);
});
