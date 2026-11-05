import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import SettingsService from "#/api/settings-service/settings-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useGitRepositories } from "#/hooks/query/use-git-repositories";
import { DEFAULT_SETTINGS } from "#/services/settings";
import type { GitRepository } from "#/types/git";
import type { Provider } from "#/types/settings";

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "cloud-ohe",
  name: "OpenHands Enterprise",
  host: "https://ohe.example.com",
  apiKey: "cloud-key",
  kind: "cloud",
};

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
};

const repo = (fullName: string, provider: Provider): GitRepository => ({
  id: fullName,
  full_name: fullName,
  git_provider: provider,
  is_public: false,
});

type RepoPages = Record<string, Record<string, GitRepository[]>>;

/**
 * Serves `/installations/search` and `/repositories/search` from fixed data.
 * `pages[installation][pageId]` holds one page; page ids chain in key order.
 */
function serveCloud(installations: string[], pages: RepoPages) {
  vi.mocked(callCloudProxy).mockImplementation((async (req: {
    path: string;
  }) => {
    const url = new URL(`https://ohe.example.com${req.path}`);
    if (url.pathname === "/api/v1/git/installations/search") {
      return { items: installations, next_page_id: null };
    }
    if (url.pathname === "/api/v1/git/repositories/search") {
      const byPage = pages[url.searchParams.get("installation_id") ?? ""];
      const pageIds = Object.keys(byPage ?? {});
      const pageId = url.searchParams.get("page_id") ?? pageIds[0];
      const next = pageIds[pageIds.indexOf(pageId) + 1] ?? null;
      return { items: byPage?.[pageId] ?? [], next_page_id: next };
    }
    throw new Error(`Unexpected cloud request ${req.path}`);
  }) as never);
}

const requestedPaths = () =>
  vi
    .mocked(callCloudProxy)
    .mock.calls.map((call) => (call[0] as { path: string }).path);

describe("useGitRepositories with cloud installations", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    vi.mocked(callCloudProxy).mockReset();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...DEFAULT_SETTINGS,
      provider_tokens_set: { azure_devops: null },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
    window.localStorage.clear();
  });

  it("lists Azure DevOps repositories from every organization, one page at a time", async () => {
    // Arrange: two organizations; org-a has two pages.
    serveCloud(["org-a", "org-b"], {
      "org-a": {
        first: [repo("org-a/proj/one", "azure_devops")],
        second: [repo("org-a/proj/two", "azure_devops")],
      },
      "org-b": { first: [repo("org-b/proj/three", "azure_devops")] },
    });

    // Act
    const { result } = renderHook(
      () => useGitRepositories({ provider: "azure_devops", pageSize: 1 }),
      { wrapper },
    );
    await waitFor(() => expect(result.current.data?.pages).toHaveLength(1));
    while (result.current.hasNextPage) {
      const loaded = result.current.data?.pages.length ?? 0;
      await act(() => result.current.fetchNextPage());
      await waitFor(() =>
        expect(result.current.data?.pages.length).toBeGreaterThan(loaded),
      );
    }

    // Assert
    const names = result.current.data?.pages.flatMap((page) =>
      page.items.map((item) => item.full_name),
    );
    expect(names).toEqual([
      "org-a/proj/one",
      "org-a/proj/two",
      "org-b/proj/three",
    ]);
    const paths = requestedPaths();
    expect(
      paths.filter((path) => path.includes("/installations/search")),
    ).toHaveLength(1);
    const repoQueries = paths
      .filter((path) => path.includes("/repositories/search"))
      .map((path) => new URL(`https://x${path}`).searchParams);
    expect(
      repoQueries.map((params) => [
        params.get("provider"),
        params.get("installation_id"),
        params.get("page_id"),
      ]),
    ).toEqual([
      ["azure_devops", "org-a", null],
      ["azure_devops", "org-a", "second"],
      ["azure_devops", "org-b", null],
    ]);
  });

  it("requests no repositories when the user has no Azure DevOps organizations", async () => {
    // Arrange
    serveCloud([], {});

    // Act
    const { result } = renderHook(
      () => useGitRepositories({ provider: "azure_devops" }),
      { wrapper },
    );
    await waitFor(() =>
      expect(requestedPaths().some((p) => p.includes("/installations"))).toBe(
        true,
      ),
    );

    // Assert: the picker falls back to its empty state.
    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(
      requestedPaths().some((path) => path.includes("/repositories/search")),
    ).toBe(false);
  });
});
