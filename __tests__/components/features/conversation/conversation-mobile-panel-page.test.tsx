import { render, screen, fireEvent, within } from "@testing-library/react";
import { afterEach, describe, it, expect, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { useConversationStore } from "#/stores/conversation-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

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

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
});

describe("ConversationMobilePanelPage", () => {
  it("identifies the panel's Cat and links to its full conversation on the owning backend", () => {
    const id = "22222222-2222-4222-8222-222222222222";
    const navigate = vi.fn();
    setRegisteredBackends([
      {
        id: "cat-backend",
        name: "Cat backend",
        host: "http://localhost:9000",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "cat-backend" });
    renderWithProviders(
      <ActiveBackendProvider>
        <ConversationMobilePanelPage
          onNavigateBack={vi.fn()}
          conversation={{
            id,
            tags: { smolpaws: "insider", insiderrole: "controller" },
            parent_conversation_id: null,
          }}
        />
      </ActiveBackendProvider>,
      { navigation: { navigate } },
    );

    const link = within(
      screen.getByTestId("conversation-mobile-panel-top"),
    ).getByRole("link", { name: "CONVERSATION$OPEN_INSIDER_CAT" });
    const expectedPath = `/extensions/insider-cat/projects/conversations/${id}?backend=cat-backend`;
    expect(link).toHaveAttribute("href", expectedPath);
    expect(within(link).getByText("CONVERSATION$INSIDER_CAT")).toBeVisible();
    fireEvent.click(link);
    expect(navigate).toHaveBeenCalledWith(expectedPath, { replace: false });
  });

  it.each([
    { tags: null, parent_conversation_id: null },
    {
      tags: { smolpaws: "insider", insiderrole: "worker" },
      parent_conversation_id: null,
    },
    {
      tags: { smolpaws: "insider", insiderrole: "controller" },
      parent_conversation_id: "11111111-1111-4111-8111-111111111111",
    },
  ])("does not label an ordinary or child panel as the Cat: %j", (identity) => {
    renderWithProviders(
      <ConversationMobilePanelPage
        onNavigateBack={vi.fn()}
        conversation={{
          id: "22222222-2222-4222-8222-222222222222",
          ...identity,
        }}
      />,
    );

    expect(screen.queryByTestId("insider-cat-badge")).not.toBeInTheDocument();
  });

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
