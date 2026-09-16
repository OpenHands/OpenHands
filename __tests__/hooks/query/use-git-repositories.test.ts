import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import GitService from "#/api/git-service/git-service.api";
import { useGitRepositories } from "#/hooks/query/use-git-repositories";
import type { RepositoryPage } from "#/types/git";

interface CapturedQueryConfig {
  queryKey: unknown[];
  queryFn(context: { pageParam: unknown }): Promise<RepositoryPage>;
  getNextPageParam(
    lastPage: RepositoryPage,
    allPages: RepositoryPage[],
    lastPageParam: unknown,
  ): unknown;
  initialPageParam: unknown;
  enabled: boolean;
  staleTime: number;
  gcTime: number;
  refetchOnWindowFocus: boolean;
}

const mocks = vi.hoisted(() => ({
  providers: undefined as string[] | undefined,
  installations: undefined as string[] | undefined,
  backend: { id: "local", kind: "local" as "local" | "cloud" },
  orgId: null as string | null,
  queryConfig: null as CapturedQueryConfig | null,
  repos: {
    data: undefined as unknown,
    isLoading: false,
    isError: false,
    hasNextPage: false,
    isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
  },
}));

vi.mock("@tanstack/react-query", () => ({
  useInfiniteQuery: (config: CapturedQueryConfig) => {
    mocks.queryConfig = config;
    return mocks.repos;
  },
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: mocks.providers }),
}));

vi.mock("#/hooks/query/use-app-installations", () => ({
  useAppInstallations: () => ({
    data:
      mocks.installations === undefined
        ? undefined
        : { items: mocks.installations, next_page_id: null },
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: mocks.backend,
    orgId: mocks.orgId,
  }),
}));

function config(): CapturedQueryConfig {
  if (!mocks.queryConfig) throw new Error("query config was not captured");
  return mocks.queryConfig;
}

const emptyPage = (nextPageId: string | null = null): RepositoryPage => ({
  items: [],
  next_page_id: nextPageId,
});

describe("useGitRepositories", () => {
  const setupMocks = () => {
    if (vi.isMockFunction(GitService.retrieveUserGitRepositories)) {
      vi.mocked(GitService.retrieveUserGitRepositories).mockRestore();
    }
    if (vi.isMockFunction(GitService.retrieveInstallationRepositories)) {
      vi.mocked(GitService.retrieveInstallationRepositories).mockRestore();
    }
    vi.clearAllMocks();
    mocks.providers = undefined;
    mocks.installations = undefined;
    mocks.backend = { id: "local", kind: "local" };
    mocks.orgId = null;
    mocks.queryConfig = null;
    mocks.repos.data = undefined;
    mocks.repos.isLoading = false;
    mocks.repos.isError = false;
    mocks.repos.hasNextPage = false;
    mocks.repos.isFetchingNextPage = false;
    mocks.repos.fetchNextPage.mockReset();
  };

  it("disables and rejects a query without a provider", async () => {
    setupMocks();
    const { result } = renderHook(() => useGitRepositories({ provider: null }));

    expect(config()).toMatchObject({
      queryKey: ["repositories", null, false, 30, "local", null],
      initialPageParam: null,
      enabled: false,
      staleTime: 300000,
      gcTime: 900000,
      refetchOnWindowFocus: false,
    });
    await expect(config().queryFn({ pageParam: null })).rejects.toThrow(
      "Provider is required",
    );
    expect(result.current).toMatchObject({
      data: undefined,
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: mocks.repos.fetchNextPage,
    });
  });

  it("loads direct user repositories with default and explicit cursors", async () => {
    setupMocks();
    mocks.providers = ["github"];
    mocks.installations = ["irrelevant-for-local-backends"];
    mocks.orgId = "org-1";
    const retrieve = vi
      .spyOn(GitService, "retrieveUserGitRepositories")
      .mockResolvedValue(emptyPage("next"));

    renderHook(() => useGitRepositories({ provider: "github", pageSize: 12 }));

    expect(config().queryKey).toEqual([
      "repositories",
      "github",
      false,
      12,
      "local",
      "org-1",
    ]);
    expect(config().enabled).toBe(true);
    expect(config().initialPageParam).toBeNull();
    await expect(config().queryFn({ pageParam: null })).resolves.toEqual(
      emptyPage("next"),
    );
    await config().queryFn({ pageParam: "cursor-2" });
    expect(retrieve).toHaveBeenNthCalledWith(1, "github", undefined, 12);
    expect(retrieve).toHaveBeenNthCalledWith(2, "github", "cursor-2", 12);
    expect(config().getNextPageParam(emptyPage("next"), [], null)).toBe("next");
    expect(config().getNextPageParam(emptyPage(), [], "next")).toBeNull();
  });

  it("loads and paginates installation repositories", async () => {
    setupMocks();
    mocks.providers = ["github"];
    mocks.installations = ["installation-a", "installation-b"];
    mocks.backend = { id: "cloud", kind: "cloud" };
    const retrieve = vi
      .spyOn(GitService, "retrieveInstallationRepositories")
      .mockResolvedValue(emptyPage("page-2"));

    renderHook(() => useGitRepositories({ provider: "github" }));

    expect(config().queryKey).toEqual([
      "repositories",
      "github",
      true,
      30,
      "cloud",
      null,
      ["installation-a", "installation-b"],
    ]);
    expect(config().initialPageParam).toEqual({
      installationIndex: 0,
      pageId: null,
    });
    await config().queryFn({
      pageParam: { installationIndex: 1, pageId: null },
    });
    await config().queryFn({
      pageParam: { installationIndex: 0, pageId: "page-2" },
    });
    expect(retrieve).toHaveBeenNthCalledWith(
      1,
      "github",
      1,
      mocks.installations,
      undefined,
      30,
    );
    expect(retrieve).toHaveBeenNthCalledWith(
      2,
      "github",
      0,
      mocks.installations,
      "page-2",
      30,
    );

    expect(
      config().getNextPageParam(emptyPage("next"), [], {
        installationIndex: 0,
        pageId: null,
      }),
    ).toEqual({ installationIndex: 0, pageId: "next" });
    expect(
      config().getNextPageParam(emptyPage(), [], {
        installationIndex: 0,
        pageId: "last",
      }),
    ).toEqual({ installationIndex: 1, pageId: null });
    expect(
      config().getNextPageParam(emptyPage(), [], {
        installationIndex: 1,
        pageId: null,
      }),
    ).toBeUndefined();
  });

  it("requires the installation list for installation-based providers", async () => {
    setupMocks();
    mocks.providers = ["bitbucket"];
    mocks.installations = undefined;

    renderHook(() => useGitRepositories({ provider: "bitbucket" }));

    expect(config().queryKey.at(-1)).toEqual([]);
    expect(config().enabled).toBe(false);
    await expect(
      config().queryFn({
        pageParam: { installationIndex: 0, pageId: null },
      }),
    ).rejects.toThrow("Missing installation list");
  });

  it.each([
    ["explicitly disabled", false, ["github"], ["installation"], false],
    ["no connected providers", true, [], ["installation"], false],
    ["empty installations", true, ["github"], [], false],
    ["ready", true, ["github"], ["installation"], true],
  ])(
    "sets enabled=false when %s",
    (_label, enabled, providers, installations, expected) => {
      setupMocks();
      mocks.backend = { id: "cloud", kind: "cloud" };
      mocks.providers = providers;
      mocks.installations = installations;

      renderHook(() => useGitRepositories({ provider: "github", enabled }));

      expect(config().enabled).toBe(expected);
    },
  );

  it("stays disabled while connected providers are unresolved", () => {
    setupMocks();
    mocks.providers = undefined;

    renderHook(() => useGitRepositories({ provider: "github" }));

    expect(config().enabled).toBe(false);
  });

  it("loads more only when another page exists and no request is active", () => {
    setupMocks();
    mocks.providers = ["github"];
    mocks.repos.hasNextPage = true;
    const { result, rerender } = renderHook(() =>
      useGitRepositories({ provider: "github" }),
    );

    act(() => result.current.onLoadMore());
    expect(mocks.repos.fetchNextPage).toHaveBeenCalledOnce();

    mocks.repos.isFetchingNextPage = true;
    rerender();
    act(() => result.current.onLoadMore());
    expect(mocks.repos.fetchNextPage).toHaveBeenCalledOnce();

    mocks.repos.hasNextPage = false;
    mocks.repos.isFetchingNextPage = false;
    rerender();
    act(() => result.current.onLoadMore());
    expect(mocks.repos.fetchNextPage).toHaveBeenCalledOnce();
  });
});
