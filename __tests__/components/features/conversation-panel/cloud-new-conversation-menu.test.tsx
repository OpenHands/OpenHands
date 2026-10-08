import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import { CloudNewConversationMenu } from "#/components/features/conversation-panel/cloud-new-conversation-menu";

const mocks = vi.hoisted(() => ({
  useGitRepositories: vi.fn(),
  useSearchRepositories: vi.fn(),
  createConversation: vi.fn(),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: (options: unknown) => {
    mocks.useGitRepositories(options);
    return {
      data: { pages: [{ items: [], next_page_id: null }] },
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
});
