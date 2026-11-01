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
  currentPath = "/plugins",
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

  it("keeps only Plugins and Apps in Customize", () => {
    renderExtensionsNavigation(<ExtensionsNavigation />);
    const nav = screen.getByTestId("extensions-navbar-desktop");
    expect(
      within(nav)
        .getAllByRole("link")
        .map((link) => link.getAttribute("href")),
    ).toEqual(["/plugins", "/apps"]);
    expect(
      within(nav).queryByTestId("sidebar-extensions-/mcp"),
    ).not.toBeInTheDocument();
    expect(
      within(nav).queryByTestId("sidebar-extensions-/skills"),
    ).not.toBeInTheDocument();
  });

  it("does not link Cloud Skills externally from Customize", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    renderExtensionsNavigation(<ExtensionsNavigation />);
    const nav = screen.getByTestId("extensions-navbar-desktop");
    expect(
      within(nav).queryByTestId("sidebar-extensions-/skills"),
    ).not.toBeInTheDocument();
  });

  describe("tablet section selector", () => {
    const originalInnerWidth = window.innerWidth;
    beforeEach(() => {
      window.innerWidth = 820;
    });
    afterEach(() => {
      window.innerWidth = originalInnerWidth;
    });

    it("navigates from Plugins to Apps", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      renderExtensionsNavigation(
        <ExtensionsCompactNavigation />,
        "/plugins",
        navigate,
      );
      const trigger = screen.getByRole("button", {
        name: "NAV$CUSTOMIZE: Plugins",
      });
      await user.click(trigger);
      const nav = screen.getByRole("navigation", { name: "NAV$CUSTOMIZE" });
      expect(
        within(nav)
          .getAllByRole("link")
          .map((link) => link.getAttribute("href")),
      ).toEqual(["/plugins", "/apps"]);
      await user.click(within(nav).getByRole("link", { name: "Apps" }));
      expect(navigate).toHaveBeenCalledWith("/apps", { replace: false });
    });

    it("supports keyboard opening, Escape focus return, and outside dismissal", async () => {
      const user = userEvent.setup();
      renderExtensionsNavigation(<ExtensionsCompactNavigation />);
      const trigger = screen.getByRole("button");
      trigger.focus();
      await user.keyboard("{Enter}{Tab}");
      expect(screen.getByRole("link", { name: "Plugins" })).toHaveFocus();
      await user.keyboard("{Escape}");
      expect(trigger).toHaveFocus();
      await user.click(trigger);
      await user.click(document.body);
      expect(trigger).toHaveAttribute("aria-expanded", "false");
    });

    it.each([767, 1024])("hides compact navigation at %i px", (width) => {
      window.innerWidth = width;
      renderExtensionsNavigation(<ExtensionsCompactNavigation />);
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
    });
  });
});
