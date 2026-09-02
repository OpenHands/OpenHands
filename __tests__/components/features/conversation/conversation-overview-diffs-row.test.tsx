import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationOverviewDiffsRow } from "#/components/features/conversation/conversation-overview-diffs-row";
import { useConversationStore } from "#/stores/conversation-store";

const navigateToTabMock = vi.fn();
const closeDrawerMock = vi.fn();

vi.mock("#/hooks/use-conversation-overview-git-diff-stats", () => ({
  useConversationOverviewGitDiffStats: () => ({
    additions: 12,
    deletions: 4,
    changeCount: 2,
    isLoading: false,
    isError: false,
  }),
}));

const navigateToChangesMock = vi.fn();

vi.mock("#/hooks/use-select-conversation-tab", () => ({
  useSelectConversationTab: () => ({
    navigateToTab: navigateToTabMock,
    navigateToChanges: navigateToChangesMock,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      git_provider: "github",
    },
  }),
}));

vi.mock(
  "#/components/features/conversation/conversation-overview-drawer-context",
  () => ({
    useConversationOverviewDrawerOptional: () => ({
      section: "skills",
      openAdd: false,
      openSection: vi.fn(),
      closeDrawer: closeDrawerMock,
    }),
  }),
);

describe("ConversationOverviewDiffsRow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationStore.setState({ messageToSend: null });
  });

  it("opens the git actions menu and sends commit, pull, push, and create PR prompts", async () => {
    const user = userEvent.setup();
    render(<ConversationOverviewDiffsRow />);

    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-action"),
    );

    await user.click(
      await screen.findByTestId("conversation-overview-diffs-git-commit"),
    );
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "commit",
    );

    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-action"),
    );
    await user.click(screen.getByTestId("conversation-overview-diffs-git-pull"));
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "pull",
    );

    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-action"),
    );
    await user.click(screen.getByTestId("conversation-overview-diffs-git-push"));
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "push",
    );

    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-action"),
    );
    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-create-pr"),
    );
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "pull request",
    );
  });

  it("keeps diff numbers hidden while the git menu is open", async () => {
    const user = userEvent.setup();
    render(<ConversationOverviewDiffsRow />);

    const stats = screen.getByTestId(
      "conversation-overview-diffs-additions",
    ).parentElement;

    await user.click(
      screen.getByTestId("conversation-overview-diffs-git-action"),
    );

    expect(stats).toHaveClass("opacity-0");
  });

  it("opens Diff view and closes open drawers when the changes label is clicked", async () => {
    const user = userEvent.setup();
    render(<ConversationOverviewDiffsRow />);

    await user.click(screen.getByTestId("conversation-overview-diffs"));

    expect(closeDrawerMock).toHaveBeenCalled();
    expect(navigateToChangesMock).toHaveBeenCalled();
    expect(navigateToTabMock).not.toHaveBeenCalled();
  });

  it("uses a full-row hover that clears when the git action is hovered", () => {
    render(<ConversationOverviewDiffsRow />);

    const row = screen.getByTestId("conversation-overview-diffs").closest("li");
    const changesButton = screen.getByTestId("conversation-overview-diffs");
    const gitAction = screen.getByTestId(
      "conversation-overview-diffs-git-action",
    );

    expect(row).toHaveClass("hover:bg-white/5");
    expect(row?.className).toContain(
      "has-[.conversation-overview-diffs-git-action:hover]:bg-transparent",
    );
    expect(changesButton).not.toHaveClass("hover:bg-white/5");
    expect(gitAction).toHaveClass("hover:bg-white/10");
  });
});
