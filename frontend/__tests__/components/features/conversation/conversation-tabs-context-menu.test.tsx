import { describe, it, expect, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";
import { ConversationTabsContextMenu } from "#/components/features/conversation/conversation-tabs/conversation-tabs-context-menu";

function renderWithRouter(conversationId: string, onClose: () => void) {
  return render(
    <MemoryRouter initialEntries={[`/conversations/${conversationId}`]}>
      <Routes>
        <Route
          path="/conversations/:conversationId"
          element={<ConversationTabsContextMenu isOpen onClose={onClose} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ConversationTabsContextMenu", () => {
  afterEach(() => {
    localStorage.clear();
  });

  it("should use per-conversation localStorage key for unpinned tabs", async () => {
    const user = userEvent.setup();
    const onClose = () => {};

    // Render for conversation-1
    const { unmount } = renderWithRouter("conversation-1", onClose);

    // Unpin the terminal tab in conversation-1
    const terminalItem = screen.getByText("COMMON$TERMINAL");
    await user.click(terminalItem);

    // Verify localStorage key is per-conversation
    const stored1 = JSON.parse(
      localStorage.getItem("conversation-unpinned-tabs-conversation-1") || "[]",
    );
    expect(stored1).toContain("terminal");

    unmount();

    // Switch to conversation-2
    renderWithRouter("conversation-2", onClose);

    // conversation-2 should have its own empty state
    const stored2 = JSON.parse(
      localStorage.getItem("conversation-unpinned-tabs-conversation-2") || "[]",
    );
    expect(stored2).toEqual([]);

    // conversation-1 state should still have terminal unpinned
    const stored1Again = JSON.parse(
      localStorage.getItem("conversation-unpinned-tabs-conversation-1") || "[]",
    );
    expect(stored1Again).toContain("terminal");
  });

  it("should toggle tab pin state when clicked", async () => {
    const user = userEvent.setup();
    const onClose = () => {};

    renderWithRouter("conversation-1", onClose);

    const terminalItem = screen.getByText("COMMON$TERMINAL");

    // Click to unpin
    await user.click(terminalItem);
    let stored = JSON.parse(
      localStorage.getItem("conversation-unpinned-tabs-conversation-1") || "[]",
    );
    expect(stored).toContain("terminal");

    // Click again to pin
    await user.click(terminalItem);
    stored = JSON.parse(
      localStorage.getItem("conversation-unpinned-tabs-conversation-1") || "[]",
    );
    expect(stored).not.toContain("terminal");
  });
});
