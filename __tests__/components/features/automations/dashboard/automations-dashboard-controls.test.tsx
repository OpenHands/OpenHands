import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { AutomationsDashboardControls } from "#/components/features/automations/dashboard/automations-dashboard-controls";
import type { DashboardSpec } from "#/manifests/automation-interface";
import { createInterfaceManifestWithSubPages } from "../../../../manifests/manifest-test-data";
import type {
  DashboardCreatedByValue,
  DashboardSortValue,
  DashboardStatusValue,
  DashboardTriggerValue,
} from "#/manifests/types";

describe("AutomationsDashboardControls — creator filter declaration", () => {
  const status: DashboardStatusValue = "all";
  const trigger: DashboardTriggerValue = "all";
  const createdBy: DashboardCreatedByValue = "all";
  const sort: DashboardSortValue = "name";
  const noop = () => {};

  function renderWithSpec(spec: DashboardSpec) {
    render(
      <AutomationsDashboardControls
        spec={spec}
        status={status}
        trigger={trigger}
        createdBy={createdBy}
        // A cloud team workspace:the host would offer the creator filter
        // whenever the manifest declares it.
        canFilterByCreator
        sort={sort}
        onStatusChange={noop}
        onTriggerChange={noop}
        onCreatedByChange={noop}
        onSortChange={noop}
      />,
    );
  }

  it("keeps the creator field hidden when the manifest does not declare it", async () => {
    // Today's shipped @openhands/extensions publishes only status and trigger
    // filters, so the admitted manifest has no created_by filter. Even on a
    // cloud team workspace, the field cannot render:there are no labels или
    // options to show, and nothing counts toward the Filters badge.

    const user = userEvent.setup();
    const list = createInterfaceManifestWithSubPages().pages.list;
    const spec: DashboardSpec = {
      ...list,
      filters: list.filters.filter((filter) => filter.id !== "created_by"),
    };
    renderWithSpec(spec);

    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );

    expect(
      within(screen.getByTestId("automations-filters-menu")).queryByTestId(
        "automations-filter-created-by",
      ),
    ).toBeNull();
    // With every filter on its default, nothing counts toward the badge and
    // Reset all stays absent.

    expect(
      within(screen.getByTestId("automations-filters-menu")).queryByTestId(
        "automations-filters-reset",
      ),
    ).toBeNull();
  });
});
