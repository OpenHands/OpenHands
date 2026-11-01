import { render, screen, fireEvent, within } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { useConversationStore } from "#/stores/conversation-store";

// Stub the heavy tab subtree so the test focuses on this page's own behavior.
vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-tabs",
  () => ({ ConversationTabs: () => <div data-testid="conversation-tabs" /> }),
);
vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-planner-build-bar",
  () => ({
    ConversationPlannerBuildBar: () => <div data-testid="planner-build-bar" />,
  }),
);
vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-tab-content/conversation-tab-content",
  () => ({ ConversationTabContent: () => <div data-testid="tab-content" /> }),
);

import { ConversationMobilePanelPage } from "#/components/features/conversation/conversation-main/conversation-mobile-panel-page";

describe("ConversationMobilePanelPage", () => {
  it("opens the right panel on mount", () => {
    render(<ConversationMobilePanelPage onNavigateBack={vi.fn()} />);

    expect(useConversationStore.getState().isRightPanelShown).toBe(true);
  });

  it("calls onNavigateBack when the back button is clicked", () => {
    const onNavigateBack = vi.fn();
    render(<ConversationMobilePanelPage onNavigateBack={onNavigateBack} />);

    fireEvent.click(screen.getByTestId("conversation-mobile-panel-back"));

    expect(onNavigateBack).toHaveBeenCalledTimes(1);
  });

  it("renders the planner build bar below the fixed-height top bar, not inside it", () => {
    render(<ConversationMobilePanelPage onNavigateBack={vi.fn()} />);

    const topBar = screen.getByTestId("conversation-mobile-panel-top");
    const buildBar = screen.getByTestId("planner-build-bar");

    expect(within(topBar).queryByTestId("planner-build-bar")).toBeNull();
    expect(
      topBar.compareDocumentPosition(buildBar) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
