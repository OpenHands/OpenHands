import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { ConversationTabsContextMenu } from "#/components/features/conversation/conversation-tabs/conversation-tabs-context-menu";
import { useConversationStore } from "#/stores/conversation-store";

const TASK_CONVERSATION_ID = "task-ec03fb2ab8604517b24af632b058c2fd";
const REAL_CONVERSATION_ID = "conv-abc123";

vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: () => false,
}));

let mockConversationId = TASK_CONVERSATION_ID;

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: mockConversationId }),
}));

const createWrapper = (conversationId: string) => {
  return ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter initialEntries={[`/conversations/${conversationId}`]}>
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe("ConversationTabs localStorage behavior", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetAllMocks();
    mockConversationId = TASK_CONVERSATION_ID;
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });
  });

  describe("task-prefixed conversation IDs", () => {
    it("should not create localStorage entries for task-prefixed conversation IDs", () => {
      render(<ConversationTabs />, {
        wrapper: createWrapper(TASK_CONVERSATION_ID),
      });

      expect(
        localStorage.getItem(`conversation-state-${TASK_CONVERSATION_ID}`),
      ).toBeNull();
    });
  });

  describe("consolidated localStorage key", () => {
    it("should use a single consolidated key for tab state", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      const changesTab = screen.getByTestId("conversation-tab-editor");
      await user.click(changesTab);

      const consolidatedKey = `conversation-state-${REAL_CONVERSATION_ID}`;
      const storedState = localStorage.getItem(consolidatedKey);
      expect(storedState).not.toBeNull();

      const parsed = JSON.parse(storedState!);
      expect(parsed).toHaveProperty("selectedTab");
      expect(parsed).toHaveProperty("rightPanelShown");
      expect(parsed).toHaveProperty("unpinnedTabs");
    });

    it("should store unpinned tabs in consolidated key via context menu", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

      const terminalItem = screen.getByText("COMMON$TERMINAL");
      await user.click(terminalItem);

      const consolidatedKey = `conversation-state-${REAL_CONVERSATION_ID}`;
      const storedState = localStorage.getItem(consolidatedKey);
      expect(storedState).not.toBeNull();

      const parsed = JSON.parse(storedState!);
      expect(parsed.unpinnedTabs).toContain("terminal");
    });
  });

  describe("hook integration", () => {
    it("should open panel and select tab when clicking a tab while panel is closed", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      // Arrange: Panel is closed, no tab selected
      useConversationStore.setState({
        selectedTab: null,
        isRightPanelShown: false,
        hasRightPanelToggled: false,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Act: Click the terminal tab
      const terminalTab = screen.getByTestId("conversation-tab-terminal");
      await user.click(terminalTab);

      // Assert: Panel should be open and terminal tab selected
      expect(useConversationStore.getState().selectedTab).toBe("terminal");
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${REAL_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe("terminal");
      expect(storedState.rightPanelShown).toBe(true);
    });

    it("should close panel when clicking the same active tab", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "editor",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Act: Click the editor tab again
      const editorTab = screen.getByTestId("conversation-tab-editor");
      await user.click(editorTab);

      // Assert: Panel should be closed
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(false);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${REAL_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.rightPanelShown).toBe(false);
    });

    it("should switch to different tab when clicking another tab while panel is open", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "editor",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Act: Click the browser tab
      const browserTab = screen.getByTestId("conversation-tab-browser");
      await user.click(browserTab);

      // Assert: Browser tab should be selected, panel still open
      expect(useConversationStore.getState().selectedTab).toBe("browser");
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${REAL_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe("browser");
    });
  });
});
