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
const observedPaths: string[] = [];

function Probe() {
  const navigation = useNavigation();
  // Captured in an effect, not during render, so the component stays pure.
  useEffect(() => {
    captured.navigation = navigation;
    observedPaths.push(navigation.getCurrentPath?.() ?? "");
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

  it("reads the committed route from a consumer's effect right after navigation", async () => {
    // The getter backs callbacks that outlive their row (a fork that settles
    // after the message unmounts). It must already report the new route by the
    // time any passive effect runs: React flushes layout effects before passive
    // ones, so a `useEffect` update of the ref would still read the previous
    // route here. A fork settling in that window would decide the reader had
    // left the source and skip navigating them into the fork.
    renderAt("/conversations/a");
    observedPaths.length = 0;

    await userEvent.click(screen.getByTestId("navigate"));

    await waitFor(() =>
      expect(captured.navigation?.getCurrentPath?.()).toBe("/conversations/b"),
    );
    // Every getter read from the consumer's effect after the click sees the new
    // route, never the stale one.
    expect(observedPaths).not.toContain("/conversations/a");
    expect(observedPaths.at(-1)).toBe("/conversations/b");
  });
});
