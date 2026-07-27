import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SettingsLayout } from "#/components/features/settings/settings-layout";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const navigationItems: SettingsNavRenderedItem[] = OSS_NAV_ITEMS.map((item) => ({
  type: "item",
  item,
}));

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
