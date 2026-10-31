import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NavigationProvider } from "#/context/navigation-context";
import {
  SettingsCompactNavigation,
  SettingsDesktopSidebar,
} from "#/components/features/settings/settings-desktop-sidebar";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SettingsLayout } from "#/components/features/settings/settings-layout";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { SettingsNavDivider } from "#/components/features/settings/settings-nav-divider";
import { SettingsNavHeader } from "#/components/features/settings/settings-nav-header";
import { I18nKey } from "#/i18n/declaration";
import { renderWithProviders } from "test-utils";
import { SettingsNavLink } from "#/components/features/settings/settings-nav-link";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { type ReactNode } from "react";
import { SettingsNavigation } from "#/components/features/settings/settings-navigation";
import { SettingsMobileHub } from "#/components/features/settings/settings-mobile-hub";

describe("settings-layout", () => {
  const navigationItems: SettingsNavRenderedItem[] = OSS_NAV_ITEMS.map(
    (item) => ({
      type: "item",
      item,
    }),
  );

  describe("SettingsLayout", () => {
    it("renders the desktop sidebar alongside the provided child content", () => {
      render(
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <ActiveBackendProvider>
            <MemoryRouter>
              <SettingsLayout navigationItems={navigationItems}>
                <div data-testid="page-body">page body</div>
              </SettingsLayout>
            </MemoryRouter>
          </ActiveBackendProvider>
        </QueryClientProvider>,
      );

      expect(screen.getByTestId("settings-navbar-desktop")).toBeInTheDocument();
      expect(screen.getByTestId("page-body")).toBeInTheDocument();
    });
  });

  describe("Settings compact section navigation", () => {
    const originalInnerWidth = window.innerWidth;
    afterEach(() => {
      window.innerWidth = originalInnerWidth;
    });

    function renderCompact(currentPath: string, items = navigationItems) {
      const navigate = vi.fn();
      const view = render(
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <ActiveBackendProvider>
            <NavigationProvider
              value={{
                currentPath,
                navigate,
                isNavigating: false,
                conversationId: null,
              }}
            >
              <SettingsCompactNavigation navigationItems={items} />
            </NavigationProvider>
          </ActiveBackendProvider>
        </QueryClientProvider>,
      );
      return { ...view, navigate };
    }

    it("uses the supplied visible sections and navigates without the hub", async () => {
      window.innerWidth = 820;
      const items = navigationItems.filter(
        (item) =>
          item.type === "item" &&
          ["/settings/app", "/settings/secrets"].includes(item.item.to),
      );
      const { navigate } = renderCompact("/settings/app", items);
      const user = userEvent.setup();

      await user.click(screen.getByRole("button"));
      const nav = screen.getByRole("navigation");
      expect(within(nav).getAllByRole("link")).toHaveLength(2);
      await user.click(
        within(nav).getByRole("link", { name: "SETTINGS$NAV_SECRETS" }),
      );

      expect(navigate).toHaveBeenCalledWith("/settings/secrets", {
        replace: false,
      });
      expect(screen.queryByRole("navigation")).not.toBeInTheDocument();
    });

    it("keeps the hub as a list without a redundant selector", () => {
      window.innerWidth = 820;
      renderCompact("/settings");
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
    });

    it.each(["/settings/llm/", "/settings/llm/profile"])(
      "keeps the LLM section reachable and selected on %s",
      async (currentPath) => {
        window.innerWidth = 820;
        renderCompact(currentPath);
        const user = userEvent.setup();

        await user.click(
          screen.getByRole("button", {
            name: "SETTINGS$TITLE: SETTINGS$NAV_LLM",
          }),
        );

        const nav = screen.getByRole("navigation");
        expect(
          within(nav).getByRole("link", { name: "SETTINGS$NAV_LLM" }),
        ).toHaveAttribute("aria-current", "page");
        expect(
          within(nav).getByRole("link", { name: "SETTINGS$NAV_APPLICATION" }),
        ).not.toHaveAttribute("aria-current");
      },
    );

    it("does not treat a similarly named path as an LLM child", () => {
      window.innerWidth = 820;
      renderCompact("/settings/llm-other");
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
    });
  });
});

describe("SettingsNavDivider", () => {
  it("should render the divider element", () => {
    // Arrange & Act
    const { container } = render(<SettingsNavDivider />);

    // Assert
    const divider = container.firstChild;
    expect(divider).toBeInTheDocument();
  });

  // Prop-passthrough contract: consumer-supplied className must land on the
  // element.
  it("should accept custom className", () => {
    // Arrange & Act
    const { container } = render(
      <SettingsNavDivider className="custom-class" />,
    );

    // Assert
    const divider = container.firstChild;
    expect(divider).toHaveClass("custom-class");
  });
});

describe("SettingsNavHeader", () => {
  it("should render the translated header text", () => {
    render(
      <SettingsNavHeader text={I18nKey.SETTINGS$PERSONAL_SETTINGS_HEADER} />,
    );

    expect(
      screen.getByText("SETTINGS$PERSONAL_SETTINGS_HEADER"),
    ).toBeInTheDocument();
  });

  // Prop-passthrough contract: consumer-supplied className must land on the
  // element.
  it("should accept custom className", () => {
    const { container } = render(
      <SettingsNavHeader
        text={I18nKey.SETTINGS$PERSONAL_SETTINGS_HEADER}
        className="custom-class"
      />,
    );

    const wrapper = container.firstChild;
    expect(wrapper).toHaveClass("custom-class");
  });
});

describe("settings-nav-link", () => {
  const mockNavItem = {
    to: "/settings/test",
    icon: <span data-testid="test-icon">Icon</span>,
    text: I18nKey.SETTINGS$NAV_API_KEYS,
    subtitle: I18nKey.SETTINGS$PAGE_LLM_SUBLINE,
  };

  const renderSettingsNavLink = (
    item = mockNavItem,
    onClick = vi.fn(),
    currentPath = "/",
  ) =>
    renderWithProviders(<SettingsNavLink item={item} onClick={onClick} />, {
      navigation: { currentPath },
    });

  describe("SettingsNavLink", () => {
    it("should render the link with icon and text", () => {
      // Arrange & Act
      renderSettingsNavLink();

      // Assert
      expect(screen.getByRole("link")).toBeInTheDocument();
      expect(screen.getByTestId("test-icon")).toBeInTheDocument();
      expect(screen.getByText("SETTINGS$NAV_API_KEYS")).toBeInTheDocument();
    });

    it("should navigate to the correct route", () => {
      // Arrange & Act
      renderSettingsNavLink();

      // Assert
      const link = screen.getByRole("link");
      expect(link).toHaveAttribute("href", "/settings/test");
    });

    it("should call onClick when clicked", async () => {
      // Arrange
      const user = userEvent.setup();
      const onClick = vi.fn();
      renderSettingsNavLink(mockNavItem, onClick);

      // Act
      await user.click(screen.getByRole("link"));

      // Assert
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("should mark the current settings route as active", () => {
      renderSettingsNavLink(mockNavItem, vi.fn(), "/settings/test");

      expect(screen.getByRole("link")).toHaveAttribute("aria-current", "page");
    });

    it("should render different text based on item prop", () => {
      // Arrange
      const customItem = {
        to: "/settings/secrets",
        icon: <span>Icon</span>,
        text: I18nKey.SETTINGS$NAV_SECRETS,
        subtitle: I18nKey.SETTINGS$PAGE_SECRETS_SUBLINE,
      };

      // Act
      renderSettingsNavLink(customItem);

      // Assert
      expect(screen.getByText("SETTINGS$NAV_SECRETS")).toBeInTheDocument();
      expect(screen.getByRole("link")).toHaveAttribute(
        "href",
        "/settings/secrets",
      );
    });
  });
});

describe("settings-navigation", () => {
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

    it("builds a half-transparent backdrop for the exported mobile drawer", async () => {
      renderSettingsNavigation(
        <SettingsNavigation
          isMobileMenuOpen
          onCloseMobileMenu={vi.fn()}
          navigationItems={baseItems}
        />,
      );
      const backdrop = screen.getByTestId("settings-navbar")
        .previousElementSibling as HTMLElement;
      const candidates = [...backdrop.classList];
      const selectors = new Set(
        candidates.map(
          (candidate) =>
            `.${candidate.replace(/[^\w-]/g, (character) => `\\${character}`)}`,
        ),
      );
      const compiler = await compile(
        await readFile(resolve("src/tailwind.css"), "utf8"),
        { base: resolve("src"), onDependency: () => {} },
      );
      const stylesheet = compiler.build(candidates);
      const backgrounds = [...stylesheet.matchAll(/(\.[^\s{]+)\s*\{([^}]+)\}/g)]
        .filter(([, selector]) => selectors.has(selector))
        .flatMap(([, , declarations]) =>
          [...declarations.matchAll(/background-color:\s*([^;]+);/g)].map(
            ([, value]) => value,
          ),
        );
      expect(backgrounds).not.toEqual([]);
      for (const background of backgrounds) {
        expect(background).toMatch(/50%|0\.5\b/);
      }
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
});
