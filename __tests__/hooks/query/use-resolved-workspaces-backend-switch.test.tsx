import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import {
  __resetActiveStoreForTests,
  getActiveBackend,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useResolvedWorkspaces } from "#/hooks/query/use-resolved-workspaces";
import { searchAllSubdirectories } from "#/hooks/query/use-search-subdirs";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";

vi.mock("#/api/workspaces-service/workspaces-service.api");
vi.mock("#/hooks/query/use-search-subdirs", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("#/hooks/query/use-search-subdirs")>();
  return { ...original, searchAllSubdirectories: vi.fn() };
});

const firstBackend: Backend = {
  id: "workspace-one",
  name: "Workspace one",
  host: "http://localhost:8101",
  apiKey: "first-key",
  kind: "local",
};

const secondBackend: Backend = {
  id: "workspace-two",
  name: "Workspace two",
  host: "http://localhost:8102",
  apiKey: "second-key",
  kind: "local",
};

describe("resolved workspace query backend isolation", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([firstBackend, secondBackend]);
    setActiveSelection({ backendId: firstBackend.id, orgId: null });
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    vi.mocked(WorkspacesService.listWorkspaces).mockImplementation(async () => {
      const backendId = getActiveBackend().backend.id;
      return {
        workspaces: [
          {
            id: `${backendId}-saved`,
            name: `${backendId} saved`,
            path: `/${backendId}/saved`,
          },
        ],
        workspaceParents: [
          { id: "shared-parent", name: "Shared", path: "/shared" },
        ],
      };
    });
    vi.mocked(searchAllSubdirectories).mockImplementation(async (path) => {
      const backendId = getActiveBackend().backend.id;
      return {
        items: [
          {
            name: `${backendId} child`,
            path: `${path}/${backendId}`,
          },
        ],
        next_page_id: null,
      };
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
    __resetActiveStoreForTests();
  });

  it("does not reuse saved or parent-derived workspaces after a backend switch", async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
    const { result } = renderHook(() => useResolvedWorkspaces(), { wrapper });

    await waitFor(() => {
      expect(
        result.current.workspaces.map((workspace) => workspace.name),
      ).toEqual(
        expect.arrayContaining(["workspace-one saved", "workspace-one child"]),
      );
    });

    act(() => {
      setActiveSelection({ backendId: secondBackend.id, orgId: null });
    });

    await waitFor(() => {
      expect(
        result.current.workspaces.map((workspace) => workspace.name),
      ).toEqual(
        expect.arrayContaining(["workspace-two saved", "workspace-two child"]),
      );
    });
    expect(
      result.current.workspaces.map((workspace) => workspace.name),
    ).not.toEqual(
      expect.arrayContaining(["workspace-one saved", "workspace-one child"]),
    );

    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all })
        .map((query) => query.queryKey),
    ).toEqual([
      LOCAL_WORKSPACES_QUERY_KEYS.byBackend(firstBackend.id, null),
      LOCAL_WORKSPACES_QUERY_KEYS.byBackend(secondBackend.id, null),
    ]);

    const sharedParentKeys = queryClient
      .getQueryCache()
      .findAll({ queryKey: ["file", "search_subdirs", "/shared"] })
      .map((query) => query.queryKey);
    expect(sharedParentKeys).toEqual([
      ["file", "search_subdirs", "/shared", firstBackend.id, null],
      ["file", "search_subdirs", "/shared", secondBackend.id, null],
    ]);
  });
});
