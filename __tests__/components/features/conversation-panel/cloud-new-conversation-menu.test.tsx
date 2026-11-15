import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import { CloudNewConversationMenu } from "#/components/features/conversation-panel/cloud-new-conversation-menu";
import GitService from "#/api/git-service/git-service.api";
import type { GitRepository } from "#/types/git";

const mocks = vi.hoisted(() => ({
  useGitRepositories: vi.fn(),
  useSearchRepositories: vi.fn(),
  createConversation: vi.fn(),
  repositories: [] as GitRepository[],
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: (options: unknown) => {
    mocks.useGitRepositories(options);
    return {
      data: { pages: [{ items: mocks.repositories, next_page_id: null }] },
      isLoading: false,
      isError: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
    };
  },
}));

vi.mock("#/hooks/query/use-search-repositories", () => ({
  useSearchRepositories: (...args: unknown[]) => {
    mocks.useSearchRepositories(...args);
    return { data: [], isLoading: false };
  },
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: ["github"], isLoadingSettings: false }),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mocks.createConversation,
    isPending: false,
  }),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

describe("CloudNewConversationMenu", () => {
  it("does not enable repository or installation queries while closed", async () => {
    renderWithProviders(
      <CloudNewConversationMenu
        popoverClassName="left-0 right-0"
        trigger={(triggerProps) => (
          <button
            type="button"
            data-testid="new-conversation-button"
            {...triggerProps}
          >
            + New conversation
          </button>
        )}
      />,
      { navigation: { currentPath: "/conversations" } },
    );

    await waitFor(() =>
      expect(mocks.useGitRepositories).toHaveBeenLastCalledWith({
        provider: "github",
        enabled: false,
      }),
    );
    expect(mocks.useSearchRepositories).toHaveBeenLastCalledWith(
      "",
      "github",
      true,
    );

    await userEvent.click(screen.getByTestId("new-conversation-button"));

    await waitFor(() =>
      expect(mocks.useGitRepositories).toHaveBeenLastCalledWith({
        provider: "github",
        enabled: true,
      }),
    );
    expect(mocks.useSearchRepositories).toHaveBeenLastCalledWith(
      "",
      "github",
      false,
    );
  });

  describe("launch branch", () => {
    const makeRepository = (
      overrides: Partial<GitRepository>,
    ): GitRepository => ({
      id: "1",
      full_name: "PROJ/app",
      git_provider: "bitbucket_data_center",
      is_public: false,
      ...overrides,
    });

    const launchRepository = async (fullName: string) => {
      renderWithProviders(
        <CloudNewConversationMenu
          popoverClassName="left-0 right-0"
          trigger={(triggerProps) => (
            <button
              type="button"
              data-testid="new-conversation-button"
              {...triggerProps}
            >
              + New conversation
            </button>
          )}
        />,
        { navigation: { currentPath: "/conversations" } },
      );
      await userEvent.click(screen.getByTestId("new-conversation-button"));
      await userEvent.click(await screen.findByText(fullName));
    };

    beforeEach(() => {
      mocks.createConversation.mockClear();
    });

    afterEach(() => {
      mocks.repositories = [];
      vi.restoreAllMocks();
    });

    it("launches on the default branch the repository list provides", async () => {
      // Arrange
      mocks.repositories = [
        makeRepository({
          full_name: "acme/app",
          git_provider: "github",
          main_branch: "develop",
        }),
      ];
      const lookup = vi.spyOn(GitService, "getRepositoryDefaultBranch");

      // Act
      await launchRepository("acme/app");

      // Assert
      await waitFor(() =>
        expect(mocks.createConversation).toHaveBeenCalledTimes(1),
      );
      expect(mocks.createConversation.mock.calls[0][0].repository).toEqual({
        name: "acme/app",
        gitProvider: "github",
        branch: "develop",
      });
      expect(lookup).not.toHaveBeenCalled();
    });

    it("looks up the default branch when the repository list has none", async () => {
      // Arrange
      mocks.repositories = [makeRepository({ main_branch: null })];
      vi.spyOn(GitService, "getRepositoryDefaultBranch").mockResolvedValue(
        "master",
      );

      // Act
      await launchRepository("PROJ/app");

      // Assert
      await waitFor(() =>
        expect(mocks.createConversation).toHaveBeenCalledTimes(1),
      );
      expect(mocks.createConversation.mock.calls[0][0].repository).toEqual({
        name: "PROJ/app",
        gitProvider: "bitbucket_data_center",
        branch: "master",
      });
    });

    it("launches without a branch when the default branch cannot be resolved", async () => {
      // Arrange
      mocks.repositories = [makeRepository({ main_branch: null })];
      vi.spyOn(GitService, "getRepositoryDefaultBranch").mockRejectedValue(
        new Error("Not Found"),
      );

      // Act
      await launchRepository("PROJ/app");

      // Assert
      await waitFor(() =>
        expect(mocks.createConversation).toHaveBeenCalledTimes(1),
      );
      expect(
        mocks.createConversation.mock.calls[0][0].repository.branch,
      ).toBeUndefined();
    });
  });
});
