import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
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
        sort={spec.sort.default}
        onStatusChange={noop}
        onTriggerChange={noop}
        onCreatedByChange={noop}
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
