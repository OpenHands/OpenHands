import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NavigationProvider } from "#/context/navigation-context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TabletCustomizeNav } from "#/components/features/skills/tablet-customize-nav";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const mockNavigate = vi.fn();

function renderTabletCustomizeNav(currentPath = "/mcp") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider
          value={{
            currentPath,
            conversationId: null,
            isNavigating: false,
            navigate: mockNavigate,
          }}
        >
          <TabletCustomizeNav />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("TabletCustomizeNav", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("does not render on /customize hub page", () => {
    renderTabletCustomizeNav("/customize");
    expect(screen.queryByTestId("tablet-sibling-nav")).not.toBeInTheDocument();
  });

  it("renders on /mcp with MCP Servers label and excludes /mcp from sibling menu", async () => {
    const user = userEvent.setup();
    renderTabletCustomizeNav("/mcp");

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    expect(trigger).toHaveTextContent("MCP Servers");

    await user.click(trigger);
    const menu = screen.getByTestId("tablet-sibling-nav-menu");
    expect(menu).toBeInTheDocument();

    expect(
      screen.queryByTestId("tablet-sibling-nav-item-/mcp"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/skills"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/plugins"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/apps"),
    ).toBeInTheDocument();
  });

  it("navigates to sibling destination when clicked", async () => {
    const user = userEvent.setup();
    renderTabletCustomizeNav("/mcp");

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    await user.click(screen.getByTestId("tablet-sibling-nav-item-/skills"));

    expect(mockNavigate).toHaveBeenCalledWith("/skills");
  });

  it("hides plugins and apps and links skills to cloud on a Cloud backend", async () => {
    const cloudBackend: Backend = {
      id: "cloud-1",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "test-token",
      kind: "cloud",
    };
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const user = userEvent.setup();
    renderTabletCustomizeNav("/mcp");

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    expect(
      screen.queryByTestId("tablet-sibling-nav-item-/plugins"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("tablet-sibling-nav-item-/apps"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId(
        "tablet-sibling-nav-item-https://app.all-hands.dev/settings/skills",
      ),
    ).toBeInTheDocument();
  });
});
