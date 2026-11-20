import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PluginsToolbar } from "#/components/features/plugins/plugins-toolbar";
import { renderWithProviders } from "test-utils";

function renderToolbar() {
  const onSearchChange = vi.fn();
  const onStatusFilterChange = vi.fn();
  renderWithProviders(
    <PluginsToolbar
      search=""
      onSearchChange={onSearchChange}
      statusFilter="all"
      onStatusFilterChange={onStatusFilterChange}
    />,
  );
  return { onSearchChange, onStatusFilterChange };
}

describe("PluginsToolbar", () => {
  it("stacks search above status filters below the md breakpoint", () => {
    renderToolbar();

    const toolbar = screen.getByTestId("plugins-toolbar");
    expect(toolbar.className).toContain("flex-col");
    expect(toolbar.className).toContain("md:flex-row");

    const search = screen.getByTestId("plugins-search-input");
    expect(search.parentElement?.className).toContain("w-full");

    const filters = screen.getByTestId("plugins-status-filter");
    const filterClasses = filters.className.split(/\s+/);
    expect(filterClasses).toContain("flex-wrap");
    expect(filterClasses).toContain("md:flex-nowrap");
    expect(filterClasses).not.toContain("shrink-0");
    expect(filterClasses).toContain("md:shrink-0");
  });

  it("keeps every status filter reachable and still filters with search", async () => {
    const user = userEvent.setup();
    const { onSearchChange, onStatusFilterChange } = renderToolbar();

    for (const value of ["all", "installed", "available", "local"]) {
      expect(screen.getByTestId(`plugins-filter-${value}`)).toBeInTheDocument();
    }

    await user.click(screen.getByTestId("plugins-filter-installed"));
    expect(onStatusFilterChange).toHaveBeenCalledWith("installed");

    await user.type(screen.getByTestId("plugins-search-input"), "git");
    expect(onSearchChange).toHaveBeenCalled();
  });
});
