import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { ConversationTabsContextMenu } from "#/components/features/conversation/conversation-tabs/conversation-tabs-context-menu";

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

      const changesTab = screen.getByText("COMMON$CHANGES");
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
});
