import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";
import { ReactRouterNavigationProvider } from "#/routes/react-router-navigation-provider";
import {
  useNavigation,
  type NavigationContextValue,
} from "#/context/navigation-context";

const captured: { navigation?: NavigationContextValue } = {};

function Probe() {
  const navigation = useNavigation();
  // Captured in an effect, not during render, so the component stays pure.
  useEffect(() => {
    captured.navigation = navigation;
  });
  return (
    <button
      data-testid="navigate"
      onClick={() => navigation.navigate("/conversations/b")}
    />
  );
}

const renderAt = (path: string) =>
  render(
    <RouterProvider
      router={createMemoryRouter(
        [
          {
            path: "*",
            Component: () => (
              <ReactRouterNavigationProvider>
                <Probe />
              </ReactRouterNavigationProvider>
            ),
          },
        ],
        { initialEntries: [path] },
      )}
    />,
  );

describe("ReactRouterNavigationProvider", () => {
  it("exposes the router's current route through both currentPath and the live getter", async () => {
    renderAt("/conversations/a");

    expect(captured.navigation?.currentPath).toBe("/conversations/a");
    expect(captured.navigation?.getCurrentPath?.()).toBe("/conversations/a");

    await userEvent.click(screen.getByTestId("navigate"));

    // The getter reads the router, not a value captured when the provider first
    // rendered, so it tracks navigation that happens after the caller unmounts.
    await waitFor(() =>
      expect(captured.navigation?.getCurrentPath?.()).toBe("/conversations/b"),
    );
    expect(captured.navigation?.currentPath).toBe("/conversations/b");
  });
});
