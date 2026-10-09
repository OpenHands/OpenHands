import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationGitActionsToggle } from "#/components/features/conversation/conversation-git-actions-toggle";
import { useConversationStore } from "#/stores/conversation-store";

const { breakpointIsMobile } = vi.hoisted(() => ({
  breakpointIsMobile: { value: false },
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => breakpointIsMobile.value,
}));

vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => false,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "conv-1",
      git_provider: "github",
    },
  }),
}));

describe("ConversationGitActionsToggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    breakpointIsMobile.value = false;
    useConversationStore.setState({ messageToSend: null });
  });

  it("stays visible on smaller screens", () => {
    breakpointIsMobile.value = true;

    render(<ConversationGitActionsToggle />);

    expect(
      screen.getByTestId("conversation-git-actions-toggle"),
    ).toBeInTheDocument();
  });

  it("opens a dropdown of git actions and fills the composer with prompts", async () => {
    const user = userEvent.setup();
    render(<ConversationGitActionsToggle />);

    await user.click(screen.getByTestId("conversation-git-actions-toggle"));

    await user.click(
      await screen.findByTestId("conversation-git-actions-commit"),
    );
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "commit",
    );

    await user.click(screen.getByTestId("conversation-git-actions-toggle"));
    await user.click(screen.getByTestId("conversation-git-actions-pull"));
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "pull",
    );

    await user.click(screen.getByTestId("conversation-git-actions-toggle"));
    await user.click(screen.getByTestId("conversation-git-actions-push"));
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "push",
    );

    await user.click(screen.getByTestId("conversation-git-actions-toggle"));
    await user.click(screen.getByTestId("conversation-git-actions-create-pr"));
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "pull request",
    );

    await user.click(screen.getByTestId("conversation-git-actions-toggle"));
    await user.click(
      screen.getByTestId("conversation-git-actions-create-new-branch"),
    );
    expect(useConversationStore.getState().messageToSend?.text).toContain(
      "new branch",
    );
  });
});
