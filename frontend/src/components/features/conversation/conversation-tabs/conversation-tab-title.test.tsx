import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ConversationTabTitle } from "#/components/features/conversation/conversation-tabs/conversation-tab-title";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: vi.fn(),
}));

const createMockHookResult = (isFetching: boolean) => ({
  data: [],
  isLoading: false,
  isFetching,
  isSuccess: true,
  isError: false,
  error: null,
  refetch: vi.fn(),
});

describe("ConversationTabTitle", () => {
  it("disables refresh button and shows loading state while fetching", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue(
      createMockHookResult(true)
    );

    render(<ConversationTabTitle title="Changes" conversationKey="editor" />);

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();

    const icon = button.querySelector("svg");
    expect(icon).toHaveClass("animate-spin");
  });

  it("enables refresh button when not fetching", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue(
      createMockHookResult(false)
    );

    render(<ConversationTabTitle title="Changes" conversationKey="editor" />);

    const button = screen.getByRole("button");
    expect(button).not.toBeDisabled();
  });
});
