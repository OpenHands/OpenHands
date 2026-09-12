import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, vi, it } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { GitRepoDropdown } from "../../../../src/components/features/home/git-repo-dropdown/git-repo-dropdown";
import { GitRepository } from "#/types/git";
import { I18nKey } from "#/i18n/declaration";

const useTranslationMock = vi.hoisted(() =>
  vi.fn((namespace: string) => ({
    t: (key: string) => `${namespace}:${key}`,
  })),
);

vi.mock("react-i18next", () => ({
  useTranslation: useTranslationMock,
}));

const translated = (key: I18nKey) => `openhands:${key}`;

// Mock the repository data hook
const mockUseRepositoryData = vi.fn();
vi.mock(
  "#/components/features/home/git-repo-dropdown/use-repository-data",
  () => ({
    useRepositoryData: (...args: unknown[]) => mockUseRepositoryData(...args),
  }),
);

// Mock the URL search hook
const mockUseUrlSearch = vi.fn();
vi.mock("#/components/features/home/git-repo-dropdown/use-url-search", () => ({
  useUrlSearch: (...args: unknown[]) => mockUseUrlSearch(...args),
}));

vi.mock("#/hooks/use-debounce", () => ({
  useDebounce: (value: string) => value,
}));

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: null }),
}));

const homeStore = vi.hoisted(() => ({
  recentRepositories: [] as GitRepository[],
}));

// Mock useHomeStore
vi.mock("#/stores/home-store", () => ({
  useHomeStore: () => homeStore,
}));

const MOCK_REPOSITORIES: GitRepository[] = [
  {
    id: "1",
    full_name: "user/repo-one",
    git_provider: "github",
    is_public: true,
  },
  {
    id: "2",
    full_name: "user/repo-two",
    git_provider: "github",
    is_public: true,
  },
  {
    id: "3",
    full_name: "org/feature-repo",
    git_provider: "github",
    is_public: false,
  },
];

const mockOnChange = vi.fn();

const setupDefaultMocks = (
  repositoryDataOverrides: Partial<
    ReturnType<typeof mockUseRepositoryData>
  > = {},
  urlSearchOverrides: Partial<ReturnType<typeof mockUseUrlSearch>> = {},
) => {
  mockUseRepositoryData.mockReturnValue({
    repositories: MOCK_REPOSITORIES,
    selectedRepository: null,
    isLoading: false,
    isError: false,
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isSearchLoading: false,
    ...repositoryDataOverrides,
  });

  mockUseUrlSearch.mockReturnValue({
    urlSearchResults: [],
    isUrlSearchLoading: false,
    ...urlSearchOverrides,
  });
};

const renderDropdown = (
  props: Partial<Parameters<typeof GitRepoDropdown>[0]> = {},
  repositoryDataOverrides: Partial<
    ReturnType<typeof mockUseRepositoryData>
  > = {},
  urlSearchOverrides: Partial<ReturnType<typeof mockUseUrlSearch>> = {},
  recentRepositories: GitRepository[] = [],
) => {
  vi.clearAllMocks();
  homeStore.recentRepositories = recentRepositories;
  // Set up mocks with optional overrides
  setupDefaultMocks(repositoryDataOverrides, urlSearchOverrides);

  return render(
    <GitRepoDropdown
      provider="github"
      onChange={mockOnChange}
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...props}
    />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: {
                  retry: false,
                },
              },
            })
          }
        >
          {children}
        </QueryClientProvider>
      ),
    },
  );
};

function setMenuScrollMetrics(
  menu: HTMLElement,
  {
    scrollTop,
    scrollHeight = 100,
    clientHeight = 20,
  }: { scrollTop: number; scrollHeight?: number; clientHeight?: number },
) {
  Object.defineProperties(menu, {
    scrollTop: { configurable: true, value: scrollTop },
    scrollHeight: { configurable: true, value: scrollHeight },
    clientHeight: { configurable: true, value: clientHeight },
  });
}

describe("GitRepoDropdown", () => {
  describe("dropdown behavior", () => {
    it("renders localized defaults and preserves presentation overrides", () => {
      const view = renderDropdown({ className: "repository-root" });
      const input = screen.getByTestId("git-repo-dropdown");

      expect(input).toHaveValue("");
      expect(input).toHaveAttribute(
        "placeholder",
        translated(I18nKey.COMMON$SEARCH_REPOSITORIES_PLACEHOLDER),
      );
      expect(input.parentElement?.parentElement).toHaveClass(
        "relative",
        "repository-root",
      );
      expect(input).toHaveClass(
        "text-inherit",
        "shadow-none",
        "pl-7",
        "pr-16",
        "text-sm",
        "font-normal",
        "leading-5",
        "placeholder:text-[var(--oh-muted)]",
        "disabled:cursor-not-allowed",
        "disabled:opacity-60",
      );
      expect(
        input.previousElementSibling?.querySelector("svg"),
      ).toBeInTheDocument();
      expect(
        input.previousElementSibling?.querySelector(".animate-spin"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).not.toBeInTheDocument();

      view.rerender(
        <GitRepoDropdown
          provider="github"
          placeholder="Choose a repository"
          className="repository-root"
          onChange={mockOnChange}
        />,
      );

      expect(input).toHaveAttribute("placeholder", "Choose a repository");
    });

    it("should open dropdown when input is clicked", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");
      await userEvent.click(input);

      // Dropdown should be open (menu should be visible)
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should keep dropdown open when clicking input while already open", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");

      // First click - open dropdown
      await userEvent.click(input);
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });

      // Second click on input - should stay open (not close)
      await userEvent.click(input);

      // Dropdown should still be open
      await waitFor(() => {
        expect(
          screen.getByTestId("git-repo-dropdown-menu"),
        ).toBeInTheDocument();
      });
    });

    it("should preserve typed text when clicking input while typing", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      // Click to open and type
      await userEvent.click(input);
      await userEvent.type(input, "repo");

      expect(input.value).toBe("repo");

      // Click on input again (should not reset text)
      await userEvent.click(input);

      // Text should be preserved
      expect(input.value).toBe("repo");
    });

    it("clears an unselected search when the menu closes", async () => {
      const user = userEvent.setup();
      renderDropdown();
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "repo");
      await user.keyboard("{Escape}");

      expect(input).toHaveValue("");
      expect(screen.getByTestId("git-repo-dropdown-menu")).toHaveClass(
        "hidden",
      );
    });

    it("preserves keyboard input when typing opens the menu", async () => {
      const user = userEvent.setup();
      renderDropdown();
      const input = screen.getByTestId("git-repo-dropdown");

      await user.tab();
      expect(input).toHaveFocus();
      await user.keyboard("repo");

      expect(input).toHaveValue("repo");
      expect(screen.getByTestId("git-repo-dropdown-menu")).not.toHaveClass(
        "hidden",
      );
    });
  });

  describe("cursor position preservation", () => {
    it("should allow editing in the middle of input text", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      // Click and type initial text
      await userEvent.click(input);
      await userEvent.type(input, "hello");

      expect(input.value).toBe("hello");

      // Move cursor to position 2 and type
      input.setSelectionRange(2, 2);
      await userEvent.type(input, "X");

      // The character should be inserted (exact position may vary based on browser behavior)
      expect(input.value).toContain("X");
    });
  });

  describe("input synchronization", () => {
    it("should show selected repository name in input when provided", async () => {
      const selectedRepository = MOCK_REPOSITORIES[0];

      renderDropdown({ value: selectedRepository.id }, { selectedRepository });

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      await waitFor(() => {
        expect(input.value).toBe(selectedRepository.full_name);
      });
    });

    it("preserves an active search when repository data changes", async () => {
      const user = userEvent.setup();
      const view = renderDropdown(
        { value: MOCK_REPOSITORIES[0].id },
        { selectedRepository: MOCK_REPOSITORIES[0] },
      );
      const input = screen.getByTestId("git-repo-dropdown");
      await waitFor(() => expect(input).toHaveValue("user/repo-one"));

      await user.click(input);
      await user.clear(input);
      await user.type(input, "draft-search");
      setupDefaultMocks({ selectedRepository: MOCK_REPOSITORIES[1] });
      view.rerender(
        <GitRepoDropdown
          provider="github"
          value={MOCK_REPOSITORIES[1].id}
          onChange={mockOnChange}
        />,
      );

      expect(input).toHaveValue("draft-search");
    });
  });

  describe("repository selection", () => {
    it("should call onChange when a repository is selected", async () => {
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown");
      await userEvent.click(input);

      // Wait for dropdown to open and show repositories
      await waitFor(() => {
        expect(screen.getByText("user/repo-one")).toBeInTheDocument();
      });

      // Click on a repository
      await userEvent.click(screen.getByText("user/repo-two"));

      expect(mockOnChange).toHaveBeenCalledWith(MOCK_REPOSITORIES[1]);
      await waitFor(() =>
        expect(screen.getByTestId("git-repo-dropdown-menu")).toHaveClass(
          "hidden",
        ),
      );
    });

    it("should keep selected repo visible even if it's not in fetched results", async () => {
      // selectedRepository from useRepositoryData stays null in our default mocks,
      // which matches the real-world scenario where a recent repo isn't yet loaded.
      renderDropdown();

      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;
      await userEvent.click(input);

      await waitFor(() => {
        expect(screen.getByText("user/repo-two")).toBeInTheDocument();
      });

      await userEvent.click(screen.getByText("user/repo-two"));

      await waitFor(() => {
        expect(input.value).toBe("user/repo-two");
      });
    });

    it("clears a locally selected repository and reports an undefined selection", async () => {
      const user = userEvent.setup();
      renderDropdown();
      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      await user.click(input);
      await user.click(await screen.findByText("user/repo-two"));
      await user.click(await screen.findByTestId("dropdown-clear"));

      await waitFor(() => expect(input).toHaveValue(""));
      expect(mockOnChange).toHaveBeenLastCalledWith(undefined);
    });

    it("still updates its visible selection when no change callback is provided", async () => {
      const user = userEvent.setup();
      renderDropdown({ onChange: undefined });
      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;

      await user.click(input);
      await user.click(await screen.findByText("user/repo-one"));

      await waitFor(() => expect(input).toHaveValue("user/repo-one"));
    });

    it("clears an externally selected repository when the controlled value becomes null", async () => {
      const selectedRepository = MOCK_REPOSITORIES[0];
      const view = renderDropdown(
        { value: selectedRepository.id },
        { selectedRepository },
      );
      const input = screen.getByTestId("git-repo-dropdown") as HTMLInputElement;
      await waitFor(() => expect(input).toHaveValue("user/repo-one"));

      setupDefaultMocks({ selectedRepository: null });
      view.rerender(
        <GitRepoDropdown
          provider="github"
          value={null}
          onChange={mockOnChange}
        />,
      );

      await waitFor(() => expect(input).toHaveValue(""));
      expect(screen.queryByTestId("dropdown-clear")).not.toBeInTheDocument();
    });

    it("marks an externally selected repository and still lists every repository", async () => {
      const user = userEvent.setup();
      renderDropdown(
        { value: MOCK_REPOSITORIES[0].id },
        { selectedRepository: MOCK_REPOSITORIES[0] },
      );
      const input = screen.getByTestId("git-repo-dropdown");
      await waitFor(() => expect(input).toHaveValue("user/repo-one"));

      await user.click(input);

      const options = screen.getAllByRole("option");
      expect(options).toHaveLength(MOCK_REPOSITORIES.length);
      const selectedOption = within(
        screen.getByTestId("git-repo-dropdown-menu"),
      ).getByRole("option", { name: "user/repo-one" });
      const unselectedOption = within(
        screen.getByTestId("git-repo-dropdown-menu"),
      ).getByRole("option", { name: "user/repo-two" });
      expect(selectedOption).toHaveAttribute("aria-selected", "true");
      expect(selectedOption).toHaveClass("bg-[var(--oh-interactive-selected)]");
      expect(unselectedOption).toHaveAttribute("aria-selected", "false");
      expect(unselectedOption).toHaveClass(
        "hover:bg-[var(--oh-interactive-hover)]",
      );
    });

    it("uses the latest callback when a selected repository is cleared", async () => {
      const user = userEvent.setup();
      const initialOnChange = vi.fn();
      const latestOnChange = vi.fn();
      const view = renderDropdown({ onChange: initialOnChange });
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.click(await screen.findByText("user/repo-one"));
      expect(initialOnChange).toHaveBeenCalledWith(MOCK_REPOSITORIES[0]);

      view.rerender(
        <GitRepoDropdown provider="github" onChange={latestOnChange} />,
      );
      await user.click(screen.getByTestId("dropdown-clear"));

      expect(latestOnChange).toHaveBeenCalledWith(undefined);
      expect(initialOnChange).toHaveBeenCalledTimes(1);
    });

    it("keeps a local selection while an external repository is unresolved", async () => {
      const user = userEvent.setup();
      const view = renderDropdown();
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.click(await screen.findByText("user/repo-two"));
      await waitFor(() => expect(input).toHaveValue("user/repo-two"));

      view.rerender(
        <GitRepoDropdown
          provider="github"
          value="unresolved-repository"
          onChange={mockOnChange}
        />,
      );

      await waitFor(() => expect(input).toHaveValue("user/repo-two"));
      expect(screen.getByTestId("dropdown-clear")).toBeInTheDocument();
    });
  });

  describe("repository search", () => {
    it.each([
      ["a repository URL", "https://github.com/user/repo-two", "user/repo-two"],
      [
        "an incomplete URL",
        "https://github.com/user",
        "https://github.com/user",
      ],
    ])(
      "passes the processed search for %s",
      async (_label, inputText, search) => {
        const user = userEvent.setup();
        renderDropdown();
        const input = screen.getByTestId("git-repo-dropdown");

        await user.click(input);
        await user.type(input, inputText);

        await waitFor(() => {
          expect(mockUseRepositoryData).toHaveBeenLastCalledWith(
            "github",
            false,
            search,
            [],
            inputText,
            undefined,
            undefined,
          );
        });
      },
    );

    it("filters a repository URL by its owner and repository path", async () => {
      const user = userEvent.setup();
      renderDropdown();
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "https://github.com/user/repo-two");

      expect(await screen.findByText("user/repo-two")).toBeInTheDocument();
      expect(screen.queryByText("user/repo-one")).not.toBeInTheDocument();
    });

    it("shows repository data directly when URL search supplied a result", async () => {
      const user = userEvent.setup();
      renderDropdown({}, {}, { urlSearchResults: [MOCK_REPOSITORIES[2]] });
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "does-not-match-any-name");

      expect(await screen.findByText("user/repo-one")).toBeInTheDocument();
      expect(screen.getByText("org/feature-repo")).toBeInTheDocument();
    });

    it("prioritizes matching recent repositories from the active provider", async () => {
      const user = userEvent.setup();
      const otherProviderRepository: GitRepository = {
        id: "gitlab-1",
        full_name: "team/gitlab-repo",
        git_provider: "gitlab",
        is_public: true,
      };
      renderDropdown({}, {}, {}, [
        MOCK_REPOSITORIES[1],
        otherProviderRepository,
      ]);

      await user.click(screen.getByTestId("git-repo-dropdown"));

      expect(
        screen.getByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).toBeInTheDocument();
      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["user/repo-two", "user/repo-one", "org/feature-repo"]);
      expect(screen.queryByText("team/gitlab-repo")).not.toBeInTheDocument();

      await user.type(screen.getByTestId("git-repo-dropdown"), "repo-two");
      expect(screen.getAllByRole("option")).toHaveLength(1);
      expect(screen.getByText("user/repo-two")).toBeInTheDocument();
    });

    it("updates and case-folds recent repositories as the search changes", async () => {
      const user = userEvent.setup();
      renderDropdown({}, {}, {}, [MOCK_REPOSITORIES[0], MOCK_REPOSITORIES[1]]);
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "REPO-TWO");

      expect(
        screen.getByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).toBeInTheDocument();
      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["user/repo-two"]);

      await user.clear(input);
      await user.type(input, "missing-repository");

      expect(
        screen.queryByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("git-repo-dropdown-empty")).toHaveTextContent(
        translated(I18nKey.HOME$NO_REPOSITORY_FOUND),
      );
    });

    it("uses a repository URL to filter recent repositories", async () => {
      const user = userEvent.setup();
      renderDropdown({}, {}, {}, [MOCK_REPOSITORIES[0], MOCK_REPOSITORIES[1]]);
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "https://github.com/user/repo-two");

      expect(
        screen.getByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).toBeInTheDocument();
      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["user/repo-two"]);
    });

    it("treats whitespace-only input as an empty search", async () => {
      const user = userEvent.setup();
      renderDropdown({}, {}, {}, [MOCK_REPOSITORIES[0]]);
      const input = screen.getByTestId("git-repo-dropdown");

      await user.click(input);
      await user.type(input, "   ");

      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["user/repo-one", "user/repo-two", "org/feature-repo"]);
      expect(
        screen.getByText(translated(I18nKey.COMMON$MOST_RECENT)),
      ).toBeInTheDocument();
    });

    it("shows the localized empty state for an empty repository list", async () => {
      const user = userEvent.setup();
      renderDropdown({}, { repositories: [] });

      await user.click(screen.getByTestId("git-repo-dropdown"));

      expect(screen.getByTestId("git-repo-dropdown-empty")).toHaveTextContent(
        translated(I18nKey.COMMON$NO_REPOSITORY),
      );
    });
  });

  describe("pagination", () => {
    it.each([
      ["near the bottom with another page", 70, true, false, 1],
      ["away from the bottom", 20, true, false, 0],
      ["near the bottom without another page", 70, false, false, 0],
      ["near the bottom during an existing fetch", 70, true, true, 0],
    ])(
      "requests the next page when scrolled %s",
      async (_label, scrollTop, hasNextPage, isFetchingNextPage, callCount) => {
        const fetchNextPage = vi.fn();
        const user = userEvent.setup();
        renderDropdown({}, { fetchNextPage, hasNextPage, isFetchingNextPage });
        await user.click(screen.getByTestId("git-repo-dropdown"));
        const menu = screen.getByTestId("git-repo-dropdown-menu");
        setMenuScrollMetrics(menu, { scrollTop });

        fireEvent.scroll(menu);

        expect(fetchNextPage).toHaveBeenCalledTimes(callCount);
      },
    );

    it("uses updated pagination state after repository data changes", async () => {
      const user = userEvent.setup();
      const initialFetchNextPage = vi.fn();
      const updatedFetchNextPage = vi.fn();
      const view = renderDropdown(
        {},
        { fetchNextPage: initialFetchNextPage, hasNextPage: false },
      );
      await user.click(screen.getByTestId("git-repo-dropdown"));

      setupDefaultMocks({
        fetchNextPage: updatedFetchNextPage,
        hasNextPage: true,
      });
      view.rerender(
        <GitRepoDropdown provider="github" onChange={mockOnChange} />,
      );
      const menu = screen.getByTestId("git-repo-dropdown-menu");
      setMenuScrollMetrics(menu, { scrollTop: 70 });
      fireEvent.scroll(menu);

      expect(updatedFetchNextPage).toHaveBeenCalledTimes(1);
      expect(initialFetchNextPage).not.toHaveBeenCalled();
    });
  });

  describe("loading feedback", () => {
    it.each([
      ["initial repositories", { isLoading: true }, {}],
      ["repository search", { isSearchLoading: true }, {}],
      ["the next page", { isFetchingNextPage: true }, {}],
      ["URL lookup", {}, { isUrlSearchLoading: true }],
    ])("shows a spinner while loading %s", (_label, repository, urlSearch) => {
      const { container } = renderDropdown({}, repository, urlSearch);

      expect(container.querySelector(".animate-spin")).toBeInTheDocument();
    });
  });
});
