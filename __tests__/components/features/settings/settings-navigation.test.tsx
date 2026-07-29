import type { ReactNode } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SettingsNavigation } from "#/components/features/settings/settings-navigation";
import { SettingsDesktopSidebar } from "#/components/features/settings/settings-desktop-sidebar";
import { SettingsMobileHub } from "#/components/features/settings/settings-mobile-hub";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const llmItem = OSS_NAV_ITEMS.find((item) => item.to === "/settings/llm")!;
const condenserItem = OSS_NAV_ITEMS.find(
  (item) => item.to === "/settings/condenser",
)!;
const verificationItem = OSS_NAV_ITEMS.find(
  (item) => item.to === "/settings/verification",
)!;
const acpAccessibleItems: SettingsNavRenderedItem[] = [
  { type: "item", item: llmItem },
  { type: "item", item: condenserItem },
  { type: "item", item: verificationItem },
];

const baseItems: SettingsNavRenderedItem[] = [
  { type: "header", text: "SETTINGS$TITLE" as never },
  { type: "item", item: llmItem },
  { type: "divider" },
  { type: "item", item: condenserItem },
];

function renderSettingsNavigation(ui: ReactNode) {
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

describe("SettingsNavigation", () => {
  it("renders the provided OSS navigation items, headers, and dividers", () => {
    renderSettingsNavigation(
      <SettingsNavigation
        isMobileMenuOpen={false}
        onCloseMobileMenu={vi.fn()}
        navigationItems={baseItems}
      />,
    );

    expect(screen.getByTestId("settings-navbar")).toBeInTheDocument();
    expect(screen.getAllByText("SETTINGS$TITLE").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SETTINGS$NAV_LLM").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText("SETTINGS$NAV_CONDENSER").length,
    ).toBeGreaterThan(0);
  });

  it("closes the mobile drawer when the close button is clicked", async () => {
    const onCloseMobileMenu = vi.fn();
    renderSettingsNavigation(
      <SettingsNavigation
        isMobileMenuOpen
        onCloseMobileMenu={onCloseMobileMenu}
        navigationItems={baseItems}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "SIDEBAR$CLOSE_MENU" }),
    );

    expect(onCloseMobileMenu).toHaveBeenCalledTimes(1);
  });

  it("closes the mobile drawer after a navigation item is selected", async () => {
    const onCloseMobileMenu = vi.fn();
    renderSettingsNavigation(
      <SettingsNavigation
        isMobileMenuOpen
        onCloseMobileMenu={onCloseMobileMenu}
        navigationItems={baseItems}
      />,
    );

    const mobileNav = screen.getByTestId("settings-navbar");
    await userEvent.click(within(mobileNav).getByText("SETTINGS$NAV_LLM"));

    expect(onCloseMobileMenu).toHaveBeenCalledTimes(1);
  });

  it("keeps ACP-accessible settings clickable in the desktop sidebar", () => {
    renderSettingsNavigation(
      <SettingsNavigation
        isMobileMenuOpen={false}
        onCloseMobileMenu={vi.fn()}
        navigationItems={acpAccessibleItems}
      />,
    );
    const desktopNav = screen.getByTestId("settings-navbar-desktop");
    for (const item of [llmItem, condenserItem, verificationItem]) {
      const link = within(desktopNav).getByTestId(
        `sidebar-settings-${item.to}`,
      );
      expect(link).toHaveAttribute("href", item.to);
      expect(link).not.toHaveAttribute("aria-disabled");
    }
  });
});

// Focused unit coverage for the two components extracted out of this file.
// The ``SettingsNavigation`` suite above already exercises the behaviors both
// surfaces share (close button, item-select dismissal, the desktop
// disabled/tooltip wiring). These cover each extracted component's distinct
// contract that the composite does not assert.
describe("SettingsDesktopSidebar", () => {
  it("renders a navigation link for each item entry and excludes headers and dividers", () => {
    // Arrange + Act: baseItems holds one header, two items, and one divider.
    renderSettingsNavigation(
      <SettingsDesktopSidebar navigationItems={baseItems} />,
    );

    // Assert: only the two ``item`` entries become links — the header and
    // divider are filtered out of the desktop rail.
    const desktopNav = screen.getByTestId("settings-navbar-desktop");
    const links = within(desktopNav).getAllByTestId(/^sidebar-settings-/);

    expect(links).toHaveLength(2);
    expect(
      within(desktopNav).getByTestId("sidebar-settings-/settings/llm"),
    ).toBeInTheDocument();
    expect(
      within(desktopNav).getByTestId("sidebar-settings-/settings/condenser"),
    ).toBeInTheDocument();
  });
});

describe("SettingsMobileHub", () => {
  it("keeps ACP-accessible settings clickable in the mobile settings hub", () => {
    renderSettingsNavigation(
      <SettingsMobileHub navigationItems={acpAccessibleItems} />,
    );

    const mobileNav = screen.getByTestId("settings-mobile-hub");
    for (const item of [llmItem, condenserItem, verificationItem]) {
      const link = within(mobileNav).getByRole("link", {
        name: item.text,
      });
      expect(link).toHaveAttribute("href", item.to);
      expect(link).not.toHaveAttribute("aria-disabled");
    }
  });
});
