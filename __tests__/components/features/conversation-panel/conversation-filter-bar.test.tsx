import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import { ConversationFilterBar } from "#/components/features/conversation-panel/conversation-filter-bar";
import type { ConversationFilterBarProps } from "#/components/features/conversation-panel/conversation-filter-bar";

function renderBar(overrides: Partial<ConversationFilterBarProps> = {}) {
  const props: ConversationFilterBarProps = {
    tagFacets: [],
    selectedTagFacets: [],
    onToggleTagFacet: vi.fn(),
    automationFacets: [],
    selectedAutomationNames: [],
    onToggleAutomationName: vi.fn(),
    onClearAll: vi.fn(),
    collapsed: false,
    onToggleCollapsed: vi.fn(),
    ...overrides,
  };
  render(<ConversationFilterBar {...props} />);
  return props;
}

describe("ConversationFilterBar", () => {
  it("renders null with no facets and no selection", () => {
    // Arrange + Act
    renderBar();

    // Assert: nothing is rendered, so the container is absent.
    expect(screen.queryByTestId("conversation-filter-bar")).toBeNull();
  });

  it("renders tag + automation chips with correct labels", () => {
    // Arrange + Act: bare tag (work=) should render as just "work", and the
    // __unnamed__ automation bucket should map to its i18n label
    // (rendered here as the missing-key string under the test's bare i18n).
    renderBar({
      tagFacets: ["work=", "owner=alice"],
      automationFacets: ["Nightly Audit", "__unnamed__"],
    });

    // Assert: chip labels are resolved.
    expect(
      screen.getByTestId("filter-chip-tag-work="),
    ).toHaveTextContent("work");
    expect(
      screen.getByTestId("filter-chip-tag-owner=alice"),
    ).toHaveTextContent("owner=alice");
    expect(
      screen.getByTestId("filter-chip-automation-Nightly Audit"),
    ).toHaveTextContent("Nightly Audit");
    expect(
      screen.getByTestId("filter-chip-automation-__unnamed__"),
    ).toHaveTextContent("CONVERSATION_PANEL$AUTOMATION_UNNAMED");
  });

  it("clicking a tag chip calls onToggleTagFacet with the raw facet", async () => {
    // Arrange
    const user = userEvent.setup();
    const props = renderBar({ tagFacets: ["owner=alice", "work="] });

    // Act
    await user.click(screen.getByTestId("filter-chip-tag-work="));

    // Assert: raw facet is passed through, label-only translation stays in the DOM.
    expect(props.onToggleTagFacet).toHaveBeenCalledTimes(1);
    expect(props.onToggleTagFacet).toHaveBeenCalledWith("work=");
  });

  it("clicking an automation chip calls onToggleAutomationName", async () => {
    // Arrange
    const user = userEvent.setup();
    const props = renderBar({
      automationFacets: ["Nightly Audit", "__unnamed__"],
    });

    // Act
    await user.click(
      screen.getByTestId("filter-chip-automation-__unnamed__"),
    );

    // Assert
    expect(props.onToggleAutomationName).toHaveBeenCalledTimes(1);
    expect(props.onToggleAutomationName).toHaveBeenCalledWith("__unnamed__");
  });

  it("shows Clear all only when something is selected and forwards to onClearAll", async () => {
    // Arrange: nothing selected -> Clear all stays hidden. A shared mock lets
    // both renders observe the same onClearAll spy.
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    renderBar({
      tagFacets: ["owner=alice"],
      automationFacets: ["Nightly Audit"],
      onClearAll,
    });
    expect(screen.queryByTestId("clear-filters-button")).toBeNull();

    // Act: re-render with a tag selection so Clear all appears, then click it.
    renderBar({
      tagFacets: ["owner=alice"],
      automationFacets: ["Nightly Audit"],
      selectedTagFacets: ["owner=alice"],
      onClearAll,
    });
    const clear = screen.getByTestId("clear-filters-button");
    await user.click(clear);

    // Assert
    expect(onClearAll).toHaveBeenCalledTimes(1);
  });

  it("selected chips reflect selected state via aria-pressed", () => {
    // Arrange + Act: a single tag is in both the facet list and the selection.
    renderBar({
      tagFacets: ["owner=alice", "work="],
      selectedTagFacets: ["owner=alice"],
    });

    // Assert: the selected chip is pressed, the unselected one isn't.
    expect(screen.getByTestId("filter-chip-tag-owner=alice")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("filter-chip-tag-work=")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("survives facets appearing after an empty first render", () => {
    // Regression: the measurement hook must run before the empty-state early
    // return, or going from zero facets to some changes the hook count and
    // React throws "Rendered more hooks than during the previous render".
    const props: ConversationFilterBarProps = {
      tagFacets: [],
      selectedTagFacets: [],
      onToggleTagFacet: vi.fn(),
      automationFacets: [],
      selectedAutomationNames: [],
      onToggleAutomationName: vi.fn(),
      onClearAll: vi.fn(),
      collapsed: false,
      onToggleCollapsed: vi.fn(),
    };
    const { rerender } = render(<ConversationFilterBar {...props} />);
    expect(screen.queryByTestId("conversation-filter-bar")).toBeNull();

    rerender(<ConversationFilterBar {...props} tagFacets={["work="]} />);
    expect(screen.getByTestId("conversation-filter-bar")).toBeInTheDocument();
  });

  describe("persisted whole-bar collapse", () => {
    // The persisted collapse tucks the whole bar behind a one-line summary
    // (narrow-screen real estate). Distinct from the ephemeral two-row clip:
    // the summary still surfaces an active-selection count so a hidden filter
    // never silently narrows the list.
    it("collapsed renders a one-line summary instead of chips", () => {
      renderBar({
        tagFacets: ["work=", "owner=alice"],
        automationFacets: ["Nightly Audit"],
        collapsed: true,
      });

      expect(screen.getByTestId("expand-filter-bar")).toBeInTheDocument();
      expect(screen.getByTestId("expand-filter-bar")).toHaveAttribute(
        "aria-expanded",
        "false",
      );
      expect(screen.queryByTestId("filter-chip-tag-work=")).toBeNull();
      // Nothing selected -> no active-count badge.
      expect(screen.queryByTestId("filter-bar-active-count")).toBeNull();
    });

    it("collapsed summary shows the active-selection count", () => {
      renderBar({
        tagFacets: ["work=", "owner=alice"],
        selectedTagFacets: ["owner=alice"],
        automationFacets: ["Nightly Audit"],
        selectedAutomationNames: ["Nightly Audit"],
        collapsed: true,
      });

      // 1 tag + 1 automation = 2 active; the mocked t drops interpolation,
      // so the assertion is on the badge's presence and key text.
      expect(screen.getByTestId("filter-bar-active-count")).toHaveTextContent(
        "CONVERSATION_PANEL$ACTIVE_FILTERS",
      );
    });

    it("collapsed with no facets and no selection still renders null", () => {
      renderBar({ collapsed: true });

      expect(screen.queryByTestId("conversation-filter-bar")).toBeNull();
    });

    it("clicking the summary calls onToggleCollapsed", async () => {
      const user = userEvent.setup();
      const props = renderBar({ tagFacets: ["work="], collapsed: true });

      await user.click(screen.getByTestId("expand-filter-bar"));

      expect(props.onToggleCollapsed).toHaveBeenCalledTimes(1);
    });

    it("expanded bar offers a collapse button that calls onToggleCollapsed", async () => {
      const user = userEvent.setup();
      const props = renderBar({ tagFacets: ["work="] });

      const collapse = screen.getByTestId("collapse-filter-bar");
      expect(collapse).toHaveAttribute(
        "aria-label",
        "CONVERSATION_PANEL$HIDE_FILTER_BAR",
      );
      await user.click(collapse);

      expect(props.onToggleCollapsed).toHaveBeenCalledTimes(1);
    });
  });
});
