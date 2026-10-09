import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NavigationProvider } from "#/context/navigation-context";
import {
  TabletSiblingNav,
  TabletSiblingNavItem,
} from "#/components/shared/tablet-sibling-nav";

const mockNavigate = vi.fn();

function renderNav(
  currentPath: string,
  currentSectionLabel: string,
  items: TabletSiblingNavItem[],
) {
  return render(
    <NavigationProvider
      value={{
        currentPath,
        conversationId: null,
        isNavigating: false,
        navigate: mockNavigate,
      }}
    >
      <TabletSiblingNav
        currentPath={currentPath}
        currentSectionLabel={currentSectionLabel}
        items={items}
      />
    </NavigationProvider>,
  );
}

const sampleItems: TabletSiblingNavItem[] = [
  { to: "/settings/agents", label: "Agent" },
  { to: "/settings/llm", label: "LLM Profiles" },
  { to: "/settings/app", label: "Application" },
  {
    href: "https://cloud.example.com/integrations",
    label: "Integrations",
    isExternal: true,
  },
];

describe("TabletSiblingNav", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders with tablet-only responsive classes and shows current section label", () => {
    renderNav("/settings/agents", "Agent", sampleItems);

    const root = screen.getByTestId("tablet-sibling-nav");
    expect(root).toHaveClass("hidden", "md:block", "lg:hidden");

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    expect(trigger).toHaveTextContent("Agent");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");
  });

  it("opens menu on click and lists sibling destinations excluding current section", async () => {
    const user = userEvent.setup();
    renderNav("/settings/agents", "Agent", sampleItems);

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    await user.click(trigger);

    expect(trigger).toHaveAttribute("aria-expanded", "true");
    const menu = screen.getByTestId("tablet-sibling-nav-menu");
    expect(menu).toBeInTheDocument();

    // Sibling items are present, but current item (/settings/agents) is excluded
    expect(
      screen.queryByTestId("tablet-sibling-nav-item-/settings/agents"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/settings/llm"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("tablet-sibling-nav-item-/settings/app"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(
        "tablet-sibling-nav-item-https://cloud.example.com/integrations",
      ),
    ).toBeInTheDocument();
  });

  it("navigates to sibling destination on item click and closes menu", async () => {
    const user = userEvent.setup();
    renderNav("/settings/agents", "Agent", sampleItems);

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    const llmItem = screen.getByTestId("tablet-sibling-nav-item-/settings/llm");
    await user.click(llmItem);

    expect(mockNavigate).toHaveBeenCalledWith("/settings/llm");
    expect(
      screen.queryByTestId("tablet-sibling-nav-menu"),
    ).not.toBeInTheDocument();
  });

  it("dismisses menu on Escape and restores focus to trigger", async () => {
    const user = userEvent.setup();
    renderNav("/settings/agents", "Agent", sampleItems);

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    await user.click(trigger);
    expect(screen.getByTestId("tablet-sibling-nav-menu")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("tablet-sibling-nav-menu"),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("supports keyboard navigation with ArrowDown and ArrowUp between choices", async () => {
    const user = userEvent.setup();
    renderNav("/settings/agents", "Agent", sampleItems);

    const trigger = screen.getByTestId("tablet-sibling-nav-trigger");
    trigger.focus();
    await user.keyboard("{ArrowDown}");

    const menu = screen.getByTestId("tablet-sibling-nav-menu");
    expect(menu).toBeInTheDocument();

    const items = within(menu).getAllByRole("menuitem");
    expect(items[0]).toHaveFocus();

    await user.keyboard("{ArrowDown}");
    expect(items[1]).toHaveFocus();

    await user.keyboard("{ArrowUp}");
    expect(items[0]).toHaveFocus();
  });

  it("renders external destination with external target and link attributes", async () => {
    const user = userEvent.setup();
    renderNav("/settings/agents", "Agent", sampleItems);

    await user.click(screen.getByTestId("tablet-sibling-nav-trigger"));
    const extItem = screen.getByTestId(
      "tablet-sibling-nav-item-https://cloud.example.com/integrations",
    );
    expect(extItem).toHaveAttribute(
      "href",
      "https://cloud.example.com/integrations",
    );
    expect(extItem).toHaveAttribute("target", "_blank");
    expect(extItem).toHaveAttribute("rel", "noopener noreferrer");
  });
});
