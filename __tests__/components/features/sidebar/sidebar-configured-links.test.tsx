import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { SidebarConfiguredLinks } from "#/components/features/sidebar/sidebar-configured-links";
import * as sidebarLinksConfig from "#/config/sidebar-links";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SidebarConfiguredLinks", () => {
  it("renders nothing when there are no configured links", () => {
    vi.spyOn(sidebarLinksConfig, "getConfiguredSidebarLinks").mockReturnValue(
      [],
    );

    const { container } = renderWithProviders(
      <SidebarConfiguredLinks collapsed={false} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders a same-origin link as in-app navigation, not a new tab", () => {
    vi.spyOn(sidebarLinksConfig, "getConfiguredSidebarLinks").mockReturnValue([
      {
        id: "dashboard",
        label: "Conversations Dashboard",
        url: `${window.location.origin}/dashboard?tab=all`,
        icon: "external-link",
      },
    ]);

    renderWithProviders(<SidebarConfiguredLinks collapsed={false} />);
    const link = screen.getByTestId("sidebar-configured-link-dashboard");

    expect(link).toHaveAttribute("href", "/dashboard?tab=all");
    expect(link).not.toHaveAttribute("target");
    expect(link).not.toHaveAttribute("rel");
    // No trailing "external" affordance for an in-app destination.
    expect(link.querySelectorAll("svg")).toHaveLength(1);
  });

  it("renders a cross-origin link as an external link that opens in a new tab", () => {
    vi.spyOn(sidebarLinksConfig, "getConfiguredSidebarLinks").mockReturnValue([
      {
        id: "enterprise",
        label: "OpenHands Enterprise",
        url: "https://openhands.dev",
        icon: "cloud",
      },
    ]);

    renderWithProviders(<SidebarConfiguredLinks collapsed={false} />);
    const link = screen.getByTestId("sidebar-configured-link-enterprise");

    expect(link).toHaveAttribute("href", "https://openhands.dev");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    // Icon slot + trailing external-link affordance.
    expect(link.querySelectorAll("svg")).toHaveLength(2);
  });
});
