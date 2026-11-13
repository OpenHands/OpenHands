import { fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider, useLocation } from "react-router";
import { describe, expect, it } from "vitest";
import { ConversationMobileBackButton } from "./conversation-mobile-back-button";
import { ReactRouterNavigationProvider } from "#/routes/react-router-navigation-provider";

function CurrentLocation() {
  const { pathname, search } = useLocation();
  return <div data-testid="current-location">{pathname + search}</div>;
}

describe("ConversationMobileBackButton", () => {
  it("returns to the conversations list and preserves backend selection", () => {
    const router = createMemoryRouter(
      [
        {
          path: "/conversations/:conversationId",
          element: (
            <ReactRouterNavigationProvider>
              <ConversationMobileBackButton />
              <CurrentLocation />
            </ReactRouterNavigationProvider>
          ),
        },
        {
          path: "/conversations",
          element: (
            <ReactRouterNavigationProvider>
              <CurrentLocation />
            </ReactRouterNavigationProvider>
          ),
        },
      ],
      {
        initialEntries: [
          "/conversations/conversation-123?backend=locked-cloud&org=org-123",
        ],
      },
    );

    render(<RouterProvider router={router} />);
    fireEvent.click(screen.getByRole("button", { name: "COMMON$BACK" }));

    expect(screen.getByTestId("current-location")).toHaveTextContent(
      "/conversations?backend=locked-cloud&org=org-123",
    );
  });
});
