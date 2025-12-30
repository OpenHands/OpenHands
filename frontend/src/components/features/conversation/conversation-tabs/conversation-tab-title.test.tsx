import { describe, it, expect, vi, beforeEach } from "vitest";

// IMPORTANT: mock BEFORE importing the component
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: vi.fn(),
}));

import { render, screen } from "@testing-library/react";
import { ConversationTabTitle } from "#/components/features/conversation/conversation-tabs/conversation-tab-title";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

const createMockHookResult = (isFetching: boolean) => ({
  data: [],
  isLoading: false,
  isFetching,
  isSuccess: true,
  isError: false,
  error: null,
  refetch: vi.fn(),
});

beforeEach(() => {
  vi.mocked(useUnifiedGetGitChanges).mockReset();
});

describe("ConversationTabTitle", () => {
  it("shows loading styles and spinner when fetching", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue(
      createMockHookResult(true),
    );

    render(<ConversationTabTitle title="Changes" conversationKey="editor" />);

    const button = screen.getByRole("button");

    // Assert visible loading styles (stable UI assertions)
    expect(button).toHaveClass("opacity-50");
    expect(button).toHaveClass("cursor-not-allowed");

    const icon = button.querySelector("svg");
    expect(icon).toHaveClass("animate-spin");
  });

  it("does not show loading styles when not fetching", () => {
    vi.mocked(useUnifiedGetGitChanges).mockReturnValue(
      createMockHookResult(false),
    );

    render(<ConversationTabTitle title="Changes" conversationKey="editor" />);

    const button = screen.getByRole("button");

    expect(button).not.toHaveClass("opacity-50");
    expect(button).not.toHaveClass("cursor-not-allowed");

    const icon = button.querySelector("svg");
    expect(icon).not.toHaveClass("animate-spin");
  });
});
