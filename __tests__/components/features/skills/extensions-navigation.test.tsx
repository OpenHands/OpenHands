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
import { useSidebarStore } from "#/stores/sidebar-store";

import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

function renderExtensionsNavigation(ui: ReactNode) {
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

describe("ExtensionsNavigation", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the MCP item as a clickable link for non-ACP agents", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    const mcpItem = within(nav).getByTestId("sidebar-extensions-/mcp");
    expect(mcpItem).not.toHaveAttribute("aria-disabled");
    // `NavigationLink` renders as <a> with an href so direct URL
    // navigation works.
    expect(mcpItem.tagName).toBe("A");
  });

  it("keeps the MCP item clickable when ACP is active", () => {
    // ACP agents now forward ``mcp_config`` to their subprocess at session
    // creation, so the MCP page is meaningful under ACP too — it is no
    // longer greyed out (unlike /settings and /settings/condenser, which
    // stay inert for ACP).
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    const mcpItem = within(nav).getByTestId("sidebar-extensions-/mcp");
    expect(mcpItem).not.toHaveAttribute("aria-disabled");
    expect(mcpItem.tagName).toBe("A");
  });

  it("leaves the Skills item clickable", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    const skillsItem = within(nav).getByTestId("sidebar-extensions-/skills");
    expect(skillsItem).not.toHaveAttribute("aria-disabled");
    expect(skillsItem.tagName).toBe("A");
  });

  it("renders the Plugins item as a live link without a Coming Soon badge", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    const pluginsItem = within(nav).getByTestId("sidebar-extensions-/plugins");
    expect(pluginsItem.tagName).toBe("A");
    expect(pluginsItem).not.toHaveAttribute("aria-disabled");
    expect(
      within(pluginsItem).queryByText("NAV$COMING_SOON"),
    ).not.toBeInTheDocument();
  });

  describe("cloud backend", () => {
    it("renders the Skills item as an external link to {cloudHost}/settings/skills with the renamed label", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderExtensionsNavigation(<ExtensionsNavigation />);

      const nav = screen.getByTestId("extensions-navbar-desktop");
      const skillsItem = within(nav).getByTestId("sidebar-extensions-/skills");
      expect(skillsItem.tagName).toBe("A");
      expect(skillsItem).toHaveAttribute(
        "href",
        "https://app.all-hands.dev/settings/skills",
      );
      expect(skillsItem).toHaveAttribute("target", "_blank");
      expect(skillsItem).toHaveAttribute("rel", "noopener noreferrer");
      expect(skillsItem).toHaveTextContent(
        "SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK",
      );
    });

    it("hides the Plugins item", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderExtensionsNavigation(<ExtensionsNavigation />);

      const nav = screen.getByTestId("extensions-navbar-desktop");
      expect(
        within(nav).queryByTestId("sidebar-extensions-/plugins"),
      ).not.toBeInTheDocument();
    });

    it("leaves the MCP Servers item as an in-app link", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderExtensionsNavigation(<ExtensionsNavigation />);

      const nav = screen.getByTestId("extensions-navbar-desktop");
      const mcpItem = within(nav).getByTestId("sidebar-extensions-/mcp");
      expect(mcpItem).not.toHaveAttribute("target");
      expect(mcpItem).toHaveAttribute("href", "/mcp");
    });
  });

  // Regression: the nav used to suppress itself at iPad-portrait widths
  // (768–1023px) whenever the primary Sidebar was expanded, leaving users
  // on /skills, /mcp, and /plugins with no way to switch between those
  // pages. It must stay rendered there, like the Settings secondary nav.
  describe("tablet viewports", () => {
    const originalInnerWidth = window.innerWidth;

    function setViewport(width: number) {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        writable: true,
        value: width,
      });
    }

    afterEach(() => {
      setViewport(originalInnerWidth);
      // The Zustand sidebar store is a module singleton — reset it so this
      // suite's state doesn't bleed into other tests.
      useSidebarStore.setState({ collapsed: false });
    });

    it("stays rendered at iPad portrait width while the Sidebar is expanded", () => {
      // Arrange: iPad Air portrait viewport with the primary Sidebar
      // expanded — the exact conditions that previously hid the nav.
      setViewport(820);
      useSidebarStore.setState({ collapsed: false });

      // Act
      renderExtensionsNavigation(<ExtensionsNavigation />);

      // Assert
      expect(
        screen.getByTestId("extensions-navbar-desktop"),
      ).toBeInTheDocument();
    });
  });
});
