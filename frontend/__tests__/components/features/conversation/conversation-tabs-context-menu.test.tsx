import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationTabsContextMenu } from "#/components/features/conversation/conversation-tabs/conversation-tabs-context-menu";

vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: () => false,
}));

const TEST_CONVERSATION_ID = "conv-123";

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: TEST_CONVERSATION_ID }),
}));

describe("ConversationTabsContextMenu localStorage behavior", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.resetAllMocks();
  });

  describe("consolidated localStorage key", () => {
    it("should store unpinned tabs in consolidated conversation-state key", async () => {
      // Desired behavior: Unpinned tabs should be stored in a single consolidated
      // conversation-state key, not in a separate conversation-unpinned-tabs key.
      const user = userEvent.setup();

      render(
        <ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />,
      );

      // Click on a tab to unpin it (e.g., terminal)
      const terminalItem = screen.getByText("COMMON$TERMINAL");
      await user.click(terminalItem);

      // The old separate keys should NOT be used
      expect(localStorage.getItem("conversation-unpinned-tabs")).toBeNull();
      expect(localStorage.getItem(`conversation-unpinned-tabs-${TEST_CONVERSATION_ID}`)).toBeNull();

      // The consolidated key SHOULD be used
      const consolidatedKey = `conversation-state-${TEST_CONVERSATION_ID}`;
      const storedState = localStorage.getItem(consolidatedKey);
      expect(storedState).not.toBeNull();

      const parsed = JSON.parse(storedState!);
      expect(parsed.unpinnedTabs).toContain("terminal");
    });
  });
});
