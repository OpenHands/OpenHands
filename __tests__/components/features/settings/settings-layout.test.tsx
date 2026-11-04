import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NavigationProvider } from "#/context/navigation-context";
import { SettingsCompactNavigation } from "#/components/features/settings/settings-desktop-sidebar";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SettingsLayout } from "#/components/features/settings/settings-layout";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

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
