import type { ReactNode } from "react";
import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { NavigationProvider } from "#/context/navigation-context";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

import {
  ExtensionsNavigation,
  ExtensionsCompactNavigation,
} from "#/components/features/skills/extensions-navigation";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string) => (key === "NAV$EXTENSIONS" ? "Apps" : key),
    i18n: { language: "en", exists: () => false },
  }),
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

function renderExtensionsNavigation(
  ui: ReactNode,
  currentPath = "/mcp",
  navigate = vi.fn(),
) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <MemoryRouter>
          <NavigationProvider
            value={{
              currentPath,
              navigate,
              isNavigating: false,
              conversationId: null,
            }}
          >
            {ui}
          </NavigationProvider>
        </MemoryRouter>
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

  it("orders MCP Servers before Skills and Plugins", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    const navigationItems = within(nav).getAllByRole("link");

    expect(navigationItems.map((item) => item.textContent)).toEqual([
      "MCP Servers",
      "Skills",
      "Plugins",
      "Apps",
    ]);
  });

  it("links Apps to its management page", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);

    const nav = screen.getByTestId("extensions-navbar-desktop");
    expect(within(nav).getByTestId("sidebar-extensions-/apps")).toHaveAttribute(
      "href",
      "/apps",
    );
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

    it("hides the Plugins and Apps items", () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });

      renderExtensionsNavigation(<ExtensionsNavigation />);

      const nav = screen.getByTestId("extensions-navbar-desktop");
      expect(
        within(nav).queryByTestId("sidebar-extensions-/plugins"),
      ).not.toBeInTheDocument();
      expect(
        within(nav).queryByTestId("sidebar-extensions-/apps"),
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

  describe("tablet section selector", () => {
    const originalInnerWidth = window.innerWidth;
    beforeEach(() => {
      window.innerWidth = 820;
    });
    afterEach(() => {
      window.innerWidth = originalInnerWidth;
    });

    it("opens every local section and navigates directly from MCP to Skills", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      renderExtensionsNavigation(
        <ExtensionsCompactNavigation />,
        "/mcp",
        navigate,
      );

      const trigger = screen.getByRole("button", {
        name: "NAV$CUSTOMIZE: MCP Servers",
      });
      await user.click(trigger);
      const nav = screen.getByRole("navigation", { name: "NAV$CUSTOMIZE" });
      expect(
        within(nav)
          .getAllByRole("link")
          .map((link) => link.getAttribute("href")),
      ).toEqual(["/mcp", "/skills", "/plugins", "/apps"]);
      expect(
        within(nav).getByRole("link", { name: "MCP Servers" }),
      ).toHaveAttribute("aria-current", "page");
      await user.click(within(nav).getByRole("link", { name: "Skills" }));

      expect(navigate).toHaveBeenCalledWith("/skills", { replace: false });
      expect(trigger).toHaveAttribute("aria-expanded", "false");
    });

    it("supports keyboard opening, Escape focus return, and outside dismissal", async () => {
      const user = userEvent.setup();
      renderExtensionsNavigation(<ExtensionsCompactNavigation />);
      const trigger = screen.getByRole("button");
      trigger.focus();
      await user.keyboard("{Enter}{Tab}");
      expect(screen.getByRole("link", { name: "MCP Servers" })).toHaveFocus();
      await user.keyboard("{Escape}");
      expect(trigger).toHaveFocus();
      expect(trigger).toHaveAttribute("aria-expanded", "false");
      await user.click(trigger);
      await user.click(document.body);
      expect(trigger).toHaveAttribute("aria-expanded", "false");
    });

    it("keeps Cloud filtering and the external Skills destination", async () => {
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      renderExtensionsNavigation(<ExtensionsCompactNavigation />);
      await userEvent.click(screen.getByRole("button"));
      const nav = screen.getByRole("navigation");
      expect(
        within(nav).queryByRole("link", { name: "Plugins" }),
      ).not.toBeInTheDocument();
      expect(
        within(nav).queryByRole("link", { name: "Apps" }),
      ).not.toBeInTheDocument();
      const link = within(nav).getByRole("link", {
        name: "SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK",
      });
      expect(link).toHaveAttribute(
        "href",
        "https://app.all-hands.dev/settings/skills",
      );
      expect(link).toHaveAttribute("target", "_blank");
    });

    it.each([767, 1024])(
      "leaves phone and desktop navigation unchanged at %i px",
      (width) => {
        window.innerWidth = width;
        renderExtensionsNavigation(<ExtensionsCompactNavigation />);
        expect(screen.queryByRole("button")).not.toBeInTheDocument();
      },
    );
  });
});
