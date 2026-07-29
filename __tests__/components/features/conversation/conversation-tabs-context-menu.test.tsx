import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationTabsContextMenu } from "#/components/features/conversation/conversation-tabs/conversation-tabs-context-menu";
import { useConversationStore } from "#/stores/conversation-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import {
  ACTIVE_BACKEND_STORAGE_KEY,
  BACKENDS_STORAGE_KEY,
} from "#/api/backend-registry/storage";
import type { Backend } from "#/api/backend-registry/types";

const CONVERSATION_ID = "conv-abc123";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: CONVERSATION_ID }),
}));

let mockHasTaskList = false;
vi.mock("#/hooks/use-task-list", () => ({
  useTaskList: () => ({
    hasTaskList: mockHasTaskList,
    taskList: [],
  }),
}));

vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => false,
}));

function seedActiveBackend(backend: Backend): void {
  localStorage.setItem(BACKENDS_STORAGE_KEY, JSON.stringify([backend]));
  localStorage.setItem(
    ACTIVE_BACKEND_STORAGE_KEY,
    JSON.stringify({ backendId: backend.id, orgId: null }),
  );
  __resetActiveStoreForTests();
}

describe("ConversationTabsContextMenu", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetActiveStoreForTests();
    mockHasTaskList = false;
    useConversationStore.setState({
      selectedTab: "files",
      isRightPanelShown: true,
      hasRightPanelToggled: true,
    });
  });

  it("should render nothing when isOpen is false", () => {
    const { container } = render(
      <ConversationTabsContextMenu isOpen={false} onClose={vi.fn()} />,
    );

    expect(container.innerHTML).toBe("");
  });

  it("should render all default tabs when open", () => {
    render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

    const expectedTabs = ["COMMON$FILES", "COMMON$TERMINAL", "COMMON$BROWSER"];
    for (const tab of expectedTabs) {
      expect(screen.getByText(tab)).toBeInTheDocument();
    }

    // Planner is cloud-only; on the default (local) backend it is hidden.
    expect(screen.queryByText("COMMON$PLANNER")).not.toBeInTheDocument();
  });

  it("should show the Planner entry when the active backend is cloud", () => {
    seedActiveBackend({
      id: "cloud-test",
      name: "Cloud Test",
      host: "https://app.example.com",
      apiKey: "secret",
      kind: "cloud",
    });

    render(
      <ActiveBackendProvider>
        <ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />
      </ActiveBackendProvider>,
    );

    expect(screen.getByText("COMMON$PLANNER")).toBeInTheDocument();
  });

  it("should open a tab from the label button without changing pin state", async () => {
    const user = userEvent.setup();

    render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

    await user.click(screen.getByTestId("conversation-tabs-menu-open-terminal"));

    expect(useConversationStore.getState().selectedTab).toBe("terminal");
    const storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${CONVERSATION_ID}`)!,
    );
    expect(storedState.unpinnedTabs).toEqual([]);
  });

  it("should re-pin a tab when clicking the pin control on an unpinned tab", async () => {
    const user = userEvent.setup();

    render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

    await user.click(screen.getByTestId("conversation-tabs-menu-pin-terminal"));
    let storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${CONVERSATION_ID}`)!,
    );
    expect(storedState.unpinnedTabs).toContain("terminal");

    await user.click(screen.getByTestId("conversation-tabs-menu-pin-terminal"));
    storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${CONVERSATION_ID}`)!,
    );
    expect(storedState.unpinnedTabs).not.toContain("terminal");
  });

  it("should switch to another pinned tab when unpinning the currently active tab via pin control", async () => {
    const user = userEvent.setup();

    render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

    expect(useConversationStore.getState().selectedTab).toBe("files");

    await user.click(screen.getByTestId("conversation-tabs-menu-pin-files"));

    const storeState = useConversationStore.getState();
    expect(storeState.hasRightPanelToggled).toBe(true);
    expect(storeState.selectedTab).toBe("terminal");

    const storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${CONVERSATION_ID}`)!,
    );
    expect(storedState.unpinnedTabs).toContain("files");
    expect(storedState.selectedTab).toBe("terminal");
  });

  it("should not close the right panel when unpinning a non-active tab", async () => {
    const user = userEvent.setup();

    render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

    await user.click(screen.getByTestId("conversation-tabs-menu-pin-terminal"));

    const storeState = useConversationStore.getState();
    expect(storeState.hasRightPanelToggled).toBe(true);
  });

  describe("with tasklist", () => {
    beforeEach(() => {
      mockHasTaskList = true;
    });

    it("should show tasklist in context menu when hasTaskList is true", () => {
      render(<ConversationTabsContextMenu isOpen={true} onClose={vi.fn()} />);

      expect(screen.getByText("COMMON$TASK_LIST")).toBeInTheDocument();
    });
  });
});
