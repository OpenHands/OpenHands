import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useBranchData } from "#/hooks/query/use-branch-data";
import type { Branch } from "#/types/git";
import type { Provider } from "#/types/settings";

const mocks = vi.hoisted(() => ({
  repositoryBranches: vi.fn(),
  searchBranches: vi.fn(),
}));

vi.mock("#/hooks/query/use-repository-branches", () => ({
  useRepositoryBranchesPaginated: mocks.repositoryBranches,
}));

vi.mock("#/hooks/query/use-search-branches", () => ({
  useSearchBranches: mocks.searchBranches,
}));

interface PaginatedBranchData {
  pages?: Array<{ items: Branch[] }>;
  pageParams: Array<string | null>;
}

interface SearchResult {
  data: Branch[] | undefined;
  isLoading: boolean;
}

interface HookProps {
  repository: string | null;
  provider: Provider;
  defaultBranch: string | null;
  processedSearchInput: string;
  inputValue: string;
  selectedBranch?: Branch | null;
}

const getBranch = (
  name: string,
  commitSha: string = `sha-${name}`,
  protectedBranch = false,
): Branch => ({
  name,
  commit_sha: commitSha,
  protected: protectedBranch,
});

const getPaginatedData = (...pages: Branch[][]): PaginatedBranchData => ({
  pages: pages.map((items) => ({ items })),
  pageParams: pages.map(() => null),
});

const getHookProps = (overrides: Partial<HookProps> = {}): HookProps => ({
  repository: "openhands/agent-canvas",
  provider: "github",
  defaultBranch: null,
  processedSearchInput: "",
  inputValue: "",
  selectedBranch: null,
  ...overrides,
});

const setupDependencies = () => {
  const fetchNextPage = vi.fn();
  const paginated = {
    data: undefined as PaginatedBranchData | undefined,
    fetchNextPage,
    hasNextPage: false,
    isLoading: false,
    isFetchingNextPage: false,
    isError: false,
  };
  const searchResults = new Map<string, SearchResult>();

  mocks.repositoryBranches.mockReset();
  mocks.searchBranches.mockReset();
  mocks.repositoryBranches.mockImplementation(() => paginated);
  mocks.searchBranches.mockImplementation(
    (_repository: string | null, query: string) =>
      searchResults.get(query) ?? {
        data: undefined,
        isLoading: false,
      },
  );

  return { fetchNextPage, paginated, searchResults };
};

const renderBranchData = (overrides: Partial<HookProps> = {}) => {
  const props = getHookProps(overrides);
  const hook = renderHook(
    (currentProps: HookProps) =>
      useBranchData(
        currentProps.repository,
        currentProps.provider,
        currentProps.defaultBranch,
        currentProps.processedSearchInput,
        currentProps.inputValue,
        currentProps.selectedBranch,
      ),
    { initialProps: props },
  );

  return { hook, props };
};

describe("useBranchData", () => {
  it("returns empty branch lists and forwards query state when data is unavailable", () => {
    const dependencies = setupDependencies();
    dependencies.paginated.hasNextPage = true;
    dependencies.paginated.isFetchingNextPage = true;
    dependencies.paginated.isError = true;
    dependencies.searchResults.set("feature", {
      data: undefined,
      isLoading: true,
    });

    const { hook } = renderBranchData({
      processedSearchInput: "feature",
      inputValue: "feature",
    });

    expect(hook.result.current).toEqual({
      branches: [],
      allBranches: [],
      fetchNextPage: dependencies.fetchNextPage,
      hasNextPage: true,
      isLoading: false,
      isFetchingNextPage: true,
      isError: true,
      isSearchLoading: true,
    });
    expect(mocks.repositoryBranches).toHaveBeenCalledWith(
      "openhands/agent-canvas",
      30,
      "github",
    );
    expect(mocks.searchBranches).toHaveBeenNthCalledWith(
      1,
      "openhands/agent-canvas",
      "feature",
      30,
      "github",
    );
    expect(mocks.searchBranches).toHaveBeenNthCalledWith(
      2,
      "openhands/agent-canvas",
      "",
      30,
      "github",
    );
  });

  it("treats branch data without a pages collection as empty", () => {
    const dependencies = setupDependencies();
    dependencies.paginated.data = { pages: undefined, pageParams: [] };

    const { hook } = renderBranchData();

    expect(hook.result.current.allBranches).toEqual([]);
    expect(hook.result.current.branches).toEqual([]);
  });

  it("updates flattened branches and default priority when pages change", () => {
    const dependencies = setupDependencies();
    const feature = getBranch("feature/coverage", "feature-sha");
    const develop = getBranch("develop", "develop-sha");
    const mainFromSearch = getBranch("main", "search-main-sha", true);
    const mainLoaded = getBranch("main", "loaded-main-sha", true);
    const duplicateMain = getBranch("main", "duplicate-main-sha", true);

    dependencies.paginated.data = getPaginatedData([feature]);
    dependencies.searchResults.set("main", {
      data: [develop, mainFromSearch],
      isLoading: true,
    });

    const { hook, props } = renderBranchData({ defaultBranch: "main" });

    expect(hook.result.current.allBranches).toEqual([feature]);
    expect(hook.result.current.branches).toEqual([mainFromSearch, feature]);
    expect(hook.result.current.isLoading).toBe(true);
    expect(hook.result.current.isSearchLoading).toBe(false);
    expect(mocks.searchBranches).toHaveBeenNthCalledWith(
      2,
      "openhands/agent-canvas",
      "main",
      30,
      "github",
    );

    dependencies.paginated.data = getPaginatedData(
      [develop],
      [mainLoaded, duplicateMain, feature],
    );
    dependencies.paginated.isLoading = true;
    hook.rerender(props);

    expect(hook.result.current.allBranches).toEqual([
      develop,
      mainLoaded,
      duplicateMain,
      feature,
    ]);
    expect(hook.result.current.branches).toEqual([
      mainLoaded,
      develop,
      feature,
    ]);
    expect(hook.result.current.isLoading).toBe(true);
    expect(mocks.searchBranches).toHaveBeenNthCalledWith(
      4,
      "openhands/agent-canvas",
      "",
      30,
      "github",
    );
  });

  it.each([
    {
      reason: "there is no default branch",
      defaultBranch: null,
      pages: [[getBranch("feature/no-default")]],
      processedSearchInput: "",
    },
    {
      reason: "the default branch is already loaded",
      defaultBranch: "main",
      pages: [[getBranch("main")]],
      processedSearchInput: "",
    },
    {
      reason: "no branches have loaded",
      defaultBranch: "main",
      pages: [],
      processedSearchInput: "",
    },
    {
      reason: "the user is searching",
      defaultBranch: "main",
      pages: [[getBranch("develop")]],
      processedSearchInput: "feature",
    },
  ])(
    "does not search for the default when $reason",
    ({ defaultBranch, pages, processedSearchInput }) => {
      const dependencies = setupDependencies();
      dependencies.paginated.data = getPaginatedData(...pages);

      renderBranchData({
        defaultBranch,
        processedSearchInput,
        inputValue: processedSearchInput,
      });

      expect(mocks.searchBranches).toHaveBeenNthCalledWith(
        2,
        "openhands/agent-canvas",
        "",
        30,
        "github",
      );
    },
  );

  it("uses active search results and prioritizes their default branch", () => {
    const dependencies = setupDependencies();
    const loadedDevelop = getBranch("develop", "loaded-develop-sha");
    const feature = getBranch("feature/search", "feature-search-sha");
    const mainFromSearch = getBranch("main", "main-search-sha", true);
    const duplicateMain = getBranch("main", "main-duplicate-sha", true);
    const release = getBranch("release", "release-sha");
    dependencies.paginated.data = getPaginatedData([loadedDevelop]);
    dependencies.searchResults.set("feature", {
      data: [feature, mainFromSearch, duplicateMain, release],
      isLoading: false,
    });

    const { hook } = renderBranchData({
      defaultBranch: "main",
      processedSearchInput: "feature",
      inputValue: "feature",
      selectedBranch: undefined,
    });

    expect(hook.result.current.allBranches).toEqual([loadedDevelop]);
    expect(hook.result.current.branches).toEqual([
      mainFromSearch,
      feature,
      release,
    ]);
  });

  it("keeps loaded branches when the input exactly matches the selection", () => {
    const dependencies = setupDependencies();
    const loadedDevelop = getBranch("develop", "loaded-develop-sha");
    const selectedFeature = getBranch("feature/exact", "selected-sha");
    const searchFeature = getBranch("feature/exact", "search-sha");
    dependencies.paginated.data = getPaginatedData([loadedDevelop]);
    dependencies.searchResults.set("feature/exact", {
      data: [searchFeature],
      isLoading: false,
    });

    const { hook } = renderBranchData({
      processedSearchInput: "feature/exact",
      inputValue: "feature/exact",
      selectedBranch: selectedFeature,
    });

    expect(hook.result.current.branches).toEqual([loadedDevelop]);
  });

  it("falls back to loaded branches while an active search has no data", () => {
    const dependencies = setupDependencies();
    const develop = getBranch("develop");
    dependencies.paginated.data = getPaginatedData([develop]);

    const { hook } = renderBranchData({
      processedSearchInput: "missing",
      inputValue: "missing",
    });

    expect(hook.result.current.branches).toEqual([develop]);
  });

  it("keeps search results when neither search source contains the default", () => {
    const dependencies = setupDependencies();
    const loadedDevelop = getBranch("develop", "loaded-develop-sha");
    const feature = getBranch("feature/only", "feature-only-sha");
    dependencies.paginated.data = getPaginatedData([loadedDevelop]);
    dependencies.searchResults.set("feature", {
      data: [feature],
      isLoading: false,
    });

    const { hook } = renderBranchData({
      defaultBranch: "main",
      processedSearchInput: "feature",
      inputValue: "feature",
    });

    expect(hook.result.current.branches).toEqual([feature]);
  });

  it.each([
    {
      state: "unavailable",
      defaultSearchData: undefined,
    },
    {
      state: "empty",
      defaultSearchData: [] as Branch[],
    },
    {
      state: "nonmatching",
      defaultSearchData: [getBranch("develop", "search-develop-sha")],
    },
  ])(
    "keeps loaded branches when default search data is $state",
    ({ defaultSearchData }) => {
      const dependencies = setupDependencies();
      const feature = getBranch("feature/loaded", "loaded-feature-sha");
      dependencies.paginated.data = getPaginatedData([feature]);
      if (defaultSearchData !== undefined) {
        dependencies.searchResults.set("main", {
          data: defaultSearchData,
          isLoading: false,
        });
      }

      const { hook } = renderBranchData({ defaultBranch: "main" });

      expect(hook.result.current.branches).toEqual([feature]);
      expect(mocks.searchBranches).toHaveBeenNthCalledWith(
        2,
        "openhands/agent-canvas",
        "main",
        30,
        "github",
      );
    },
  );
});
