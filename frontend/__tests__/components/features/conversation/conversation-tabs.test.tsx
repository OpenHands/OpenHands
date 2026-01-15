import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";

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
      // Issue: V1 conversations start with task-{uuid} in URL, creating localStorage entries
      // that become orphaned when navigating to the real conversation ID.
      // Desired behavior: Skip localStorage persistence for task-* IDs entirely.

      render(<ConversationTabs />, { wrapper: createWrapper(TASK_CONVERSATION_ID) });

      // No localStorage entries should be created for task-prefixed IDs
      // Using consolidated key pattern
      expect(
        localStorage.getItem(`conversation-state-${TASK_CONVERSATION_ID}`),
      ).toBeNull();

      // Old individual keys should also not exist
      expect(
        localStorage.getItem(
          `conversation-selected-tab-${TASK_CONVERSATION_ID}`,
        ),
      ).toBeNull();
      expect(
        localStorage.getItem(
          `conversation-right-panel-shown-${TASK_CONVERSATION_ID}`,
        ),
      ).toBeNull();
      expect(
        localStorage.getItem(
          `conversation-unpinned-tabs-${TASK_CONVERSATION_ID}`,
        ),
      ).toBeNull();
    });
  });

  describe("consolidated localStorage key", () => {
    it("should use a single consolidated key for all conversation state", async () => {
      // Desired behavior: All conversation state should be stored in one key
      // instead of multiple separate keys.
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      render(<ConversationTabs />, { wrapper: createWrapper(REAL_CONVERSATION_ID) });

      // Click a tab to trigger a state change that persists to localStorage
      // The Changes tab has visible text, so use that
      const changesTab = screen.getByText("COMMON$CHANGES");
      await user.click(changesTab);

      // Should use consolidated key
      const consolidatedKey = `conversation-state-${REAL_CONVERSATION_ID}`;
      const storedState = localStorage.getItem(consolidatedKey);
      expect(storedState).not.toBeNull();

      const parsed = JSON.parse(storedState!);
      expect(parsed).toHaveProperty("selectedTab");
      expect(parsed).toHaveProperty("rightPanelShown");
      expect(parsed).toHaveProperty("unpinnedTabs");

      // Old individual keys should NOT exist
      expect(
        localStorage.getItem(`conversation-selected-tab-${REAL_CONVERSATION_ID}`),
      ).toBeNull();
      expect(
        localStorage.getItem(`conversation-right-panel-shown-${REAL_CONVERSATION_ID}`),
      ).toBeNull();
      expect(
        localStorage.getItem(`conversation-unpinned-tabs-${REAL_CONVERSATION_ID}`),
      ).toBeNull();
    });
  });
});
