import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useLocalWorkspaces } from "./use-local-workspaces";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "./query-keys";

describe("useLocalWorkspaces", () => {
  it("scopes its cache entry to the active backend", async () => {
    vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    } as never);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
    const { result } = renderHook(() => useLocalWorkspaces(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const [key] = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
    // Unscoped, the picker offers paths that only exist on another server.
    expect(key.slice(0, LOCAL_WORKSPACES_QUERY_KEYS.all.length)).toEqual([
      ...LOCAL_WORKSPACES_QUERY_KEYS.all,
    ]);
    expect(key.length).toBe(LOCAL_WORKSPACES_QUERY_KEYS.all.length + 2);
  });
});
