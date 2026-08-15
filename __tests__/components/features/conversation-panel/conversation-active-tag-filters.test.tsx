import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { ConversationActiveTagFilters } from "#/components/features/conversation-panel/conversation-active-tag-filters";

describe("ConversationActiveTagFilters", () => {
  it("renders nothing when no filter is active", () => {
    renderWithProviders(
      <ConversationActiveTagFilters
        selectedFacets={[]}
        onToggleFacet={vi.fn()}
        onClearAll={vi.fn()}
      />,
    );

    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();
  });

  it("names every active facet so a narrowed list is never unexplained", () => {
    renderWithProviders(
      <ConversationActiveTagFilters
        selectedFacets={["project=vault", "work"]}
        onToggleFacet={vi.fn()}
        onClearAll={vi.fn()}
      />,
    );

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
    renderWithProviders(
      <ConversationActiveTagFilters
        selectedFacets={["project=vault", "work"]}
        onToggleFacet={onToggleFacet}
        onClearAll={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("active-tag-filter-project=vault"));

    expect(onToggleFacet).toHaveBeenCalledWith("project=vault");
  });

  it("clears every filter at once", async () => {
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    renderWithProviders(
      <ConversationActiveTagFilters
        selectedFacets={["project=vault"]}
        onToggleFacet={vi.fn()}
        onClearAll={onClearAll}
      />,
    );

    await user.click(screen.getByTestId("clear-tag-filters"));

    expect(onClearAll).toHaveBeenCalledTimes(1);
  });
});
