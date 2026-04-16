import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import { RecentConversations } from "#/components/features/home/recent-conversations/recent-conversations";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

const renderRecentConversations = () => {
  const RouterStub = createRoutesStub([
    {
      Component: () => <RecentConversations />,
      path: "/",
    },
  ]);

  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(<RouterStub />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
};

describe("RecentConversations", () => {
  const searchConversationsSpy = vi.spyOn(
    V1ConversationService,
    "searchConversations",
  );

  it("should not show empty state when there is an error", async () => {
    searchConversationsSpy.mockRejectedValue(
      new Error("Failed to fetch conversations"),
    );

    renderRecentConversations();

    // Wait for the error to be displayed
    await waitFor(() => {
      expect(
        screen.getByText("Failed to fetch conversations"),
      ).toBeInTheDocument();
    });

    // The empty state should NOT be displayed when there's an error
    expect(
      screen.queryByText("HOME$NO_RECENT_CONVERSATIONS"),
    ).not.toBeInTheDocument();
  });
});
