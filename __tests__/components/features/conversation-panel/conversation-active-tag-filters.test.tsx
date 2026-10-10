import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationActiveTagFilters } from "#/components/features/conversation-panel/conversation-active-tag-filters";
import { UNNAMED_AUTOMATION_FACET } from "#/components/features/conversation-panel/conversation-panel-list-helpers";

const renderStrip = (
  props: Partial<
    React.ComponentProps<typeof ConversationActiveTagFilters>
  > = {},
) =>
  renderWithProviders(
    <ConversationActiveTagFilters
      selectedFacets={[]}
      onToggleFacet={vi.fn()}
      selectedAutomationNames={[]}
      onToggleAutomationName={vi.fn()}
      onClearAll={vi.fn()}
      {...props}
    />,
  );

describe("ConversationActiveTagFilters", () => {
  it("renders nothing when no filter is active", () => {
    renderStrip();

    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();
  });

  it("names every active facet so a narrowed list is never unexplained", () => {
    renderStrip({ selectedFacets: ["project=vault", "work"] });

    expect(
      screen.getByTestId("conversation-active-tag-filters"),
    ).toBeInTheDocument();
    // Keyed and bare tags both read the way the facet rows label them.
    expect(
      screen.getByTestId("active-tag-filter-project=vault"),
    ).toHaveTextContent("project=vault");
    expect(screen.getByTestId("active-tag-filter-work")).toHaveTextContent(
      "work",
    );
  });

  it("drops a single filter from the strip itself", async () => {
    const user = userEvent.setup();
    const onToggleFacet = vi.fn();
    renderStrip({ selectedFacets: ["project=vault", "work"], onToggleFacet });

    await user.click(screen.getByTestId("active-tag-filter-project=vault"));

    expect(onToggleFacet).toHaveBeenCalledWith("project=vault");
  });

  it("clears every filter at once", async () => {
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    renderStrip({ selectedFacets: ["project=vault"], onClearAll });

    await user.click(screen.getByTestId("clear-tag-filters"));

    expect(onClearAll).toHaveBeenCalledTimes(1);
  });

  it("names an active automation-name filter too, so it is never the invisible narrowing", () => {
    // The automation-name selection is persisted and its rows live inside the
    // advanced-options modal; without a chip here a reload silently hides
    // conversations with nothing on screen to say why.
    renderStrip({ selectedAutomationNames: ["Nightly Audit"] });

    expect(
      screen.getByTestId("conversation-active-tag-filters"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("active-automation-filter-Nightly Audit"),
    ).toHaveTextContent("Nightly Audit");
  });

  it("labels the unnamed-automation bucket rather than leaking its sentinel", () => {
    renderStrip({ selectedAutomationNames: [UNNAMED_AUTOMATION_FACET] });

    expect(
      screen.getByTestId(
        `active-automation-filter-${UNNAMED_AUTOMATION_FACET}`,
      ),
    ).not.toHaveTextContent(UNNAMED_AUTOMATION_FACET);
  });

  it("drops a single automation-name filter from the strip", async () => {
    const user = userEvent.setup();
    const onToggleAutomationName = vi.fn();
    renderStrip({
      selectedAutomationNames: ["Nightly Audit"],
      onToggleAutomationName,
    });

    await user.click(
      screen.getByTestId("active-automation-filter-Nightly Audit"),
    );

    expect(onToggleAutomationName).toHaveBeenCalledWith("Nightly Audit");
  });

  it("shows both families side by side", () => {
    renderStrip({
      selectedFacets: ["project=vault"],
      selectedAutomationNames: ["Nightly Audit"],
    });

    expect(
      screen.getByTestId("active-tag-filter-project=vault"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("active-automation-filter-Nightly Audit"),
    ).toBeInTheDocument();
  });
});
