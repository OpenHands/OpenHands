import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationsDashboardControls } from "#/components/features/automations/dashboard/automations-dashboard-controls";
import type { DashboardSpec } from "#/manifests/automation-interface";
import { createInterfaceManifestWithSubPages } from "../../../../manifests/manifest-test-data";

describe("AutomationsDashboardControls — creator filter declaration", () => {
  const noop = () => {};

  it("hides the creator field and its value when the manifest does not declare it", async () => {
    // Arrange — a cloud team workspace, which offers the creator filter
    // whenever the manifest declares it, and a manifest without created_by
    // (as in a package before the filter was added). The creator value is
    // not the default, so a badge or Reset all would come only from it.
    const user = userEvent.setup();
    const list = createInterfaceManifestWithSubPages().pages.list;
    const spec: DashboardSpec = {
      overview: list.overview!,
      filters: list.filters!.filter((filter) => filter.id !== "created_by"),
      sort: list.sort!,
      insights: list.insights!,
    };
    render(
      <AutomationsDashboardControls
        spec={spec}
        status="all"
        trigger="all"
        createdBy="me"
        canFilterByCreator
        tag={null}
        tagOptions={[]}
        sort={spec.sort.default}
        onStatusChange={noop}
        onTriggerChange={noop}
        onCreatedByChange={noop}
        onTagChange={noop}
        onSortChange={noop}
      />,
    );
    const filtersButton = within(
      screen.getByTestId("automations-filters"),
    ).getByTestId("dropdown-trigger");

    // Act
    await user.click(filtersButton);

    // Assert — no field, no active-filter badge, no Reset all.
    const menu = screen.getByTestId("automations-filters-menu");
    expect(
      within(menu).queryByTestId("automations-filter-created-by"),
    ).not.toBeInTheDocument();
    expect(
      within(menu).queryByTestId("automations-filters-reset"),
    ).not.toBeInTheDocument();
    expect(within(filtersButton).queryByText(/^\d+$/)).not.toBeInTheDocument();
  });
});

describe("AutomationsDashboardControls — tag filter", () => {
  const noop = () => {};
  const list = createInterfaceManifestWithSubPages().pages.list;
  const spec: DashboardSpec = {
    overview: list.overview!,
    filters: [
      ...list.filters!,
      { id: "tag", label: "Filter widgets by tag", allLabel: "Any tag" },
    ],
    sort: list.sort!,
    insights: list.insights!,
  };

  function renderControls(
    props: {
      tag?: string | null;
      tagOptions?: string[];
      onTagChange?: (value: string | null) => void;
    } = {},
  ) {
    render(
      <AutomationsDashboardControls
        spec={spec}
        status="all"
        trigger="all"
        createdBy="all"
        canFilterByCreator={false}
        tag={props.tag ?? null}
        tagOptions={props.tagOptions ?? ["software-factory", "ci"]}
        sort={spec.sort.default}
        onStatusChange={noop}
        onTriggerChange={noop}
        onCreatedByChange={noop}
        onTagChange={props.onTagChange ?? noop}
        onSortChange={noop}
      />,
    );
    return within(screen.getByTestId("automations-filters")).getByTestId(
      "dropdown-trigger",
    );
  }

  it("offers the org's tags after the manifest's neutral option", async () => {
    // Arrange
    const user = userEvent.setup();
    const onTagChange = vi.fn();
    const filtersButton = renderControls({ onTagChange });
    await user.click(filtersButton);
    const field = screen.getByTestId("automations-filter-tag");

    // Act
    await user.click(within(field).getByTestId("dropdown-trigger"));
    const options = within(
      screen.getByTestId("automations-filter-tag-menu"),
    ).getAllByRole("menuitemradio");
    await user.click(
      screen.getByTestId("automations-filter-tag-software-factory"),
    );

    // Assert
    expect(options.map((option) => option.textContent)).toEqual([
      "Any tag",
      "software-factory",
      "ci",
    ]);
    expect(onTagChange).toHaveBeenCalledWith("software-factory");
  });

  it("counts a selected tag as active and resets it to any tag", async () => {
    // Arrange
    const user = userEvent.setup();
    const onTagChange = vi.fn();
    const filtersButton = renderControls({ tag: "ci", onTagChange });

    // Act
    await user.click(filtersButton);
    await user.click(screen.getByTestId("automations-filters-reset"));

    // Assert
    expect(within(filtersButton).getByText("1")).toBeInTheDocument();
    expect(onTagChange).toHaveBeenCalledWith(null);
  });

  it("hides the tag filter while the org has no tags", async () => {
    // Arrange
    const user = userEvent.setup();
    const filtersButton = renderControls({ tagOptions: [] });

    // Act
    await user.click(filtersButton);

    // Assert
    expect(
      screen.queryByTestId("automations-filter-tag"),
    ).not.toBeInTheDocument();
  });
});
