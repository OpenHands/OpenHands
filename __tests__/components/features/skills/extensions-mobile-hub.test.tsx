import type { ReactNode } from "react";
import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ExtensionsMobileHub } from "#/components/features/skills/extensions-mobile-hub";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

function renderMobileHub(ui: ReactNode) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <MemoryRouter>{ui}</MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("ExtensionsMobileHub", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the Plugins item as an enabled link", () => {
    renderMobileHub(<ExtensionsMobileHub />);

    const hub = screen.getByTestId("extensions-mobile-hub");
    const pluginsItem = within(hub).getByTestId("sidebar-extensions-/plugins");
    expect(pluginsItem).not.toHaveAttribute("aria-disabled");
  });

  it("links Apps to its management page", () => {
    renderMobileHub(<ExtensionsMobileHub />);

    const hub = screen.getByTestId("extensions-mobile-hub");
    expect(within(hub).getByTestId("sidebar-extensions-/apps")).toHaveAttribute(
      "href",
      "/apps",
    );
  });

  it("keeps agent-related items out of the mobile Customize navigation", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    renderMobileHub(<ExtensionsMobileHub />);
    const hub = screen.getByTestId("extensions-mobile-hub");
    expect(
      within(hub).queryByTestId("sidebar-extensions-/skills"),
    ).not.toBeInTheDocument();
    expect(
      within(hub).queryByTestId("sidebar-extensions-/mcp"),
    ).not.toBeInTheDocument();
  });
});
