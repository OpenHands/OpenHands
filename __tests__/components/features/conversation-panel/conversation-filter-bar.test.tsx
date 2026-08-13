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
    };
    const { rerender } = render(<ConversationFilterBar {...props} />);
    expect(screen.queryByTestId("conversation-filter-bar")).toBeNull();

    rerender(<ConversationFilterBar {...props} tagFacets={["work="]} />);
    expect(screen.getByTestId("conversation-filter-bar")).toBeInTheDocument();
  });
});
