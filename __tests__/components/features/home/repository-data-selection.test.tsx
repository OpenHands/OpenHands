import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useRepositoryData } from "#/components/features/home/git-repo-dropdown/use-repository-data";
import type { GitRepository } from "#/types/git";
import type { Provider } from "#/types/settings";

const useGitRepositoriesMock = vi.fn();
const useSearchRepositoriesMock = vi.fn();

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: (options: unknown) => useGitRepositoriesMock(options),
}));

vi.mock("#/hooks/query/use-search-repositories", () => ({
  useSearchRepositories: (...args: unknown[]) =>
    useSearchRepositoriesMock(...args),
}));

type RepositoryDataProps = {
  provider: Provider;
  disabled: boolean;
  processedSearchInput: string;
  urlSearchResults: GitRepository[];
  inputValue: string;
  value?: string | null;
  repositoryName?: string | null;
};

type RepositoryData = {
  pages?: Array<{ items: GitRepository[] }>;
};

type GitRepositoryState = {
  data: RepositoryData | undefined;
  fetchNextPage: ReturnType<typeof vi.fn>;
  hasNextPage: boolean;
  isLoading: boolean;
  isFetchingNextPage: boolean;
  isError: boolean;
};

type SearchRepositoryState = {
  data: GitRepository[] | undefined;
  isLoading: boolean;
};

type HarnessOptions = {
  props?: Partial<RepositoryDataProps>;
  git?: Partial<GitRepositoryState>;
  search?: Partial<SearchRepositoryState>;
};

const createRepository = (
  id: string,
  fullName = `owner/${id}`,
): GitRepository => ({
  id,
  full_name: fullName,
  git_provider: "github",
  is_public: true,
});

const createRepositories = (count: number): GitRepository[] =>
  Array.from({ length: count }, (_, index) =>
    createRepository(`repo-${index + 1}`),
  );

const createProps = (
  overrides: Partial<RepositoryDataProps> = {},
): RepositoryDataProps => ({
  provider: "github",
  disabled: false,
  processedSearchInput: "",
  urlSearchResults: [],
  inputValue: "",
  value: undefined,
  repositoryName: undefined,
  ...overrides,
});

const createGitState = (
  overrides: Partial<GitRepositoryState> = {},
): GitRepositoryState => ({
  data: { pages: [{ items: createRepositories(2) }] },
  fetchNextPage: vi.fn(),
  hasNextPage: false,
  isLoading: false,
  isFetchingNextPage: false,
  isError: false,
  ...overrides,
});

const createSearchState = (
  overrides: Partial<SearchRepositoryState> = {},
): SearchRepositoryState => ({
  data: undefined,
  isLoading: false,
  ...overrides,
});

const renderRepositoryData = (options: HarnessOptions = {}) => {
  const initialProps = createProps(options.props);
  const fetchNextPage = options.git?.fetchNextPage ?? vi.fn();
  const gitState = {
    current: createGitState({
      ...options.git,
      fetchNextPage,
    }),
  };
  const searchState = {
    current: createSearchState(options.search),
  };

  useGitRepositoriesMock.mockImplementation(() => gitState.current);
  useSearchRepositoriesMock.mockImplementation(() => searchState.current);

  const rendered = renderHook(
    (props: RepositoryDataProps) =>
      useRepositoryData(
        props.provider,
        props.disabled,
        props.processedSearchInput,
        props.urlSearchResults,
        props.inputValue,
        props.value,
        props.repositoryName,
      ),
    { initialProps },
  );

  return {
    ...rendered,
    initialProps,
    fetchNextPage,
    gitState,
    searchState,
  };
};

afterEach(() => {
  vi.resetAllMocks();
});

describe("repository data selection", () => {
  it("enables repository loading and skips only a search that repeats the selected name", () => {
    const { initialProps, rerender } = renderRepositoryData({
      props: {
        disabled: true,
        processedSearchInput: "owner/selected",
        inputValue: "owner/selected",
        repositoryName: "owner/selected",
      },
    });

    expect(useGitRepositoriesMock).toHaveBeenLastCalledWith({
      provider: "github",
      enabled: false,
    });
    expect(useSearchRepositoriesMock).toHaveBeenLastCalledWith(
      "owner/selected",
      "github",
      true,
    );

    rerender({
      ...initialProps,
      disabled: false,
      repositoryName: "owner/different",
    });

    expect(useGitRepositoriesMock).toHaveBeenLastCalledWith({
      provider: "github",
      enabled: true,
    });
    expect(useSearchRepositoriesMock).toHaveBeenLastCalledWith(
      "owner/selected",
      "github",
      false,
    );
  });

  it("flattens paginated repositories and forwards repository query state", () => {
    const first = createRepository("first");
    const second = createRepository("second");
    const third = createRepository("third");
    const fetchNextPage = vi.fn();
    const { result } = renderRepositoryData({
      props: { processedSearchInput: "active-search" },
      git: {
        data: {
          pages: [{ items: [first, second] }, { items: [third] }],
        },
        fetchNextPage,
        hasNextPage: true,
        isLoading: true,
        isFetchingNextPage: true,
        isError: true,
      },
      search: { isLoading: true },
    });

    expect(result.current).toMatchObject({
      repositories: [first, second, third],
      allRepositories: [first, second, third],
      selectedRepository: null,
      fetchNextPage,
      hasNextPage: true,
      isLoading: true,
      isFetchingNextPage: true,
      isError: true,
      isSearchLoading: true,
    });
  });

  it.each([
    ["repository data is unavailable", undefined],
    ["repository pages are unavailable", {}],
  ])("returns an empty collection when %s", (_reason, data) => {
    const { result } = renderRepositoryData({
      props: { value: "missing" },
      git: { data },
    });

    expect(result.current.allRepositories).toEqual([]);
    expect(result.current.repositories).toEqual([]);
    expect(result.current.selectedRepository).toBeNull();
  });

  it("treats an empty selected value as no repository", () => {
    const repositoryWithEmptyId = createRepository("", "owner/empty-id");
    const { result } = renderRepositoryData({
      props: { value: "" },
      git: { data: { pages: [{ items: [repositoryWithEmptyId] }] } },
    });

    expect(result.current.selectedRepository).toBeNull();
  });

  it.each([
    {
      source: "paginated repositories",
      createOptions: () => {
        const selected = createRepository("selected", "paged/selected");
        return {
          selected,
          options: {
            props: {
              value: selected.id,
              urlSearchResults: [createRepository(selected.id, "url/selected")],
            },
            git: { data: { pages: [{ items: [selected] }] } },
            search: {
              data: [createRepository(selected.id, "search/selected")],
            },
          } satisfies HarnessOptions,
        };
      },
    },
    {
      source: "URL search results",
      createOptions: () => {
        const selected = createRepository("selected", "url/selected");
        return {
          selected,
          options: {
            props: { value: selected.id, urlSearchResults: [selected] },
            git: { data: { pages: [{ items: [] }] } },
            search: {
              data: [createRepository("different", "search/different")],
            },
          } satisfies HarnessOptions,
        };
      },
    },
    {
      source: "regular search results",
      createOptions: () => {
        const selected = createRepository("selected", "search/selected");
        return {
          selected,
          options: {
            props: { value: selected.id },
            git: { data: { pages: [{ items: [] }] } },
            search: { data: [selected] },
          } satisfies HarnessOptions,
        };
      },
    },
  ])("resolves the selected repository from $source", ({ createOptions }) => {
    const { selected, options } = createOptions();
    const { result } = renderRepositoryData(options);

    expect(result.current.selectedRepository).toBe(selected);
  });

  it("prioritizes URL search results for display", () => {
    const paginated = createRepository("paginated");
    const searched = createRepository("searched");
    const fromUrl = createRepository("url");
    const { result } = renderRepositoryData({
      props: {
        processedSearchInput: "search",
        urlSearchResults: [fromUrl],
        inputValue: "search",
      },
      git: { data: { pages: [{ items: [paginated] }] } },
      search: { data: [searched] },
    });

    expect(result.current.repositories).toEqual([fromUrl]);
  });

  it("displays regular search results while the input differs from any selection", () => {
    const paginated = createRepository("paginated");
    const searched = createRepository("searched");
    const { result } = renderRepositoryData({
      props: {
        processedSearchInput: "search",
        inputValue: "search",
      },
      git: { data: { pages: [{ items: [paginated] }] } },
      search: { data: [searched] },
    });

    expect(result.current.selectedRepository).toBeNull();
    expect(result.current.repositories).toEqual([searched]);
  });

  it("displays an empty search result rather than stale paginated repositories", () => {
    const { result } = renderRepositoryData({
      props: {
        processedSearchInput: "nothing-matches",
        inputValue: "nothing-matches",
      },
      search: { data: [] },
    });

    expect(result.current.repositories).toEqual([]);
  });

  it("keeps paginated repositories visible when the search query is empty", () => {
    const paginated = createRepository("paginated");
    const searched = createRepository("searched");
    const { result } = renderRepositoryData({
      git: { data: { pages: [{ items: [paginated] }] } },
      search: { data: [searched] },
    });

    expect(result.current.repositories).toEqual([paginated]);
  });

  it("keeps paginated repositories visible while search data is unavailable", () => {
    const paginated = createRepository("paginated");
    const { result } = renderRepositoryData({
      props: {
        processedSearchInput: "pending-search",
        inputValue: "pending-search",
      },
      git: { data: { pages: [{ items: [paginated] }] } },
      search: { data: undefined },
    });

    expect(result.current.repositories).toEqual([paginated]);
  });

  it("keeps paginated repositories visible when the input is the selected repository name", () => {
    const selected = createRepository("selected", "owner/selected");
    const searched = createRepository("searched");
    const { result } = renderRepositoryData({
      props: {
        processedSearchInput: "owner/selected",
        inputValue: selected.full_name,
        value: selected.id,
      },
      git: { data: { pages: [{ items: [selected] }] } },
      search: { data: [searched] },
    });

    expect(result.current.selectedRepository).toBe(selected);
    expect(result.current.repositories).toEqual([selected]);
  });

  it("automatically loads another page when fewer than ten repositories are available", () => {
    const { fetchNextPage } = renderRepositoryData({
      git: {
        data: { pages: [{ items: createRepositories(9) }] },
        hasNextPage: true,
      },
    });

    expect(fetchNextPage).toHaveBeenCalledTimes(1);
  });

  it.each([
    {
      reason: "the dropdown is disabled",
      createOptions: (): HarnessOptions => ({ props: { disabled: true } }),
    },
    {
      reason: "repositories are initially loading",
      createOptions: (): HarnessOptions => ({ git: { isLoading: true } }),
    },
    {
      reason: "the next page is already loading",
      createOptions: (): HarnessOptions => ({
        git: { isFetchingNextPage: true },
      }),
    },
    {
      reason: "search results are loading",
      createOptions: (): HarnessOptions => ({
        search: { isLoading: true },
      }),
    },
    {
      reason: "there is no next page",
      createOptions: (): HarnessOptions => ({ git: { hasNextPage: false } }),
    },
    {
      reason: "a text search is active",
      createOptions: (): HarnessOptions => ({
        props: {
          processedSearchInput: "search",
          inputValue: "search",
        },
      }),
    },
    {
      reason: "URL search results are displayed",
      createOptions: (): HarnessOptions => ({
        props: { urlSearchResults: [createRepository("url-result")] },
      }),
    },
    {
      reason: "no repositories have loaded",
      createOptions: (): HarnessOptions => ({
        git: { data: { pages: [{ items: [] }] } },
      }),
    },
    {
      reason: "ten repositories already make the menu scrollable",
      createOptions: (): HarnessOptions => ({
        git: { data: { pages: [{ items: createRepositories(10) }] } },
      }),
    },
  ])("does not auto-load when $reason", ({ createOptions }) => {
    const options = createOptions();
    const { fetchNextPage } = renderRepositoryData({
      ...options,
      git: {
        data: { pages: [{ items: createRepositories(2) }] },
        hasNextPage: true,
        ...options.git,
      },
    });

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it("re-evaluates derived repositories when query results and inputs change", () => {
    const initial = createRepository("initial");
    const next = createRepository("next");
    const fromUrl = createRepository("url");
    const searched = createRepository("searched");
    const { result, rerender, initialProps, gitState, searchState } =
      renderRepositoryData({
        props: { value: initial.id },
        git: { data: { pages: [{ items: [initial] }] } },
      });

    expect(result.current.allRepositories).toEqual([initial]);
    expect(result.current.selectedRepository).toBe(initial);
    expect(result.current.repositories).toEqual([initial]);

    gitState.current = createGitState({
      data: { pages: [{ items: [next] }] },
    });
    searchState.current = createSearchState({ data: [searched] });
    rerender({
      ...initialProps,
      processedSearchInput: "url",
      urlSearchResults: [fromUrl],
      inputValue: "url",
      value: fromUrl.id,
    });

    expect(result.current.allRepositories).toEqual([next]);
    expect(result.current.selectedRepository).toBe(fromUrl);
    expect(result.current.repositories).toEqual([fromUrl]);
  });

  it("auto-loads when a rerender makes the repository list eligible", () => {
    const { fetchNextPage, initialProps, rerender } = renderRepositoryData({
      props: { disabled: true },
      git: { hasNextPage: true },
    });

    expect(fetchNextPage).not.toHaveBeenCalled();

    rerender({ ...initialProps, disabled: false });

    expect(fetchNextPage).toHaveBeenCalledTimes(1);
  });
});
