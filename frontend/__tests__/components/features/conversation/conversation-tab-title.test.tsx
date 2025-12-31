import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, afterEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationTabTitle } from "#/components/features/conversation/conversation-tabs/conversation-tab-title";

const mockRefetch = vi.fn();

vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => ({
    refetch: mockRefetch,
    isFetching: false,
  }),
}));

describe("ConversationTabTitle", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows refresh button for editor tab", () => {
    renderWithProviders(
      <ConversationTabTitle title="Changes" conversationKey="editor" />,
    );

    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("calls refetch when refresh button is clicked", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <ConversationTabTitle title="Changes" conversationKey="editor" />,
    );

    await user.click(screen.getByRole("button"));
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });
});
