import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

// IMPORTANT: mock the hook BEFORE importing the component
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: vi.fn(),
}));

import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { ConversationTabTitle } from "#/components/features/conversation/conversation-tabs/conversation-tab-title";

describe("ConversationTabTitle", () => {
  it("disables refresh button and shows loading state while fetching", () => {
    // Mock fetching state
    (useUnifiedGetGitChanges as any).mockReturnValue({
      refetch: vi.fn(),
      isFetching: true,
    });

    render(
      <ConversationTabTitle title="Changes" conversationKey="editor" />
    );

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();

    const icon = button.querySelector("svg");
    expect(icon?.classList.contains("animate-spin")).toBe(true);
  });

  it("enables refresh button when not fetching", () => {
    // Mock idle state
    (useUnifiedGetGitChanges as any).mockReturnValue({
      refetch: vi.fn(),
      isFetching: false,
    });

    render(
      <ConversationTabTitle title="Changes" conversationKey="editor" />
    );

    const button = screen.getByRole("button");
    expect(button).not.toBeDisabled();
  });
});
