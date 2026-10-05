import { render, screen } from "@testing-library/react";
import { useRef } from "react";
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
  useOptionalConversationId: () => ({ conversationId: CONVERSATION_ID }),
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

const MENU_WIDTH = 220;
const MENU_HEIGHT = 300;

/**
 * Renders the menu against a trigger pinned to a fixed viewport rect so the
 * portal's collision handling can be asserted. jsdom reports a 1024x768
 * viewport and zero-sized elements, so both rects are stubbed.
 */
function AnchoredMenu({
  top,
  bottom,
  left,
}: {
  top: number;
  bottom: number;
  left: number;
}) {
  const anchorRef = useRef<HTMLButtonElement>(null);

  const setAnchorRect = (node: HTMLButtonElement | null) => {
    anchorRef.current = node;
    if (!node) return;
    vi.spyOn(node, "getBoundingClientRect").mockReturnValue({
      top,
      bottom,
      left,
      right: left + 24,
      width: 24,
      height: bottom - top,
      x: left,
      y: top,
      toJSON: () => ({}),
    } as DOMRect);
  };

  return (
    <>
      <button type="button" ref={setAnchorRect} />
      <ConversationTabsContextMenu
        isOpen
        onClose={vi.fn()}
        anchorRef={anchorRef}
      />
    </>
  );
}

const renderAnchoredMenu = (rect: {
  top: number;
  bottom: number;
  left: number;
}) => {
  vi.spyOn(HTMLUListElement.prototype, "getBoundingClientRect").mockReturnValue(
    {
      width: MENU_WIDTH,
      height: MENU_HEIGHT,
      top: 0,
      bottom: MENU_HEIGHT,
      left: 0,
      right: MENU_WIDTH,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect,
  );

  render(<AnchoredMenu {...rect} />);
  return document.querySelector<HTMLElement>('div[style*="position: fixed"]');
};

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

    const expectedTabs = [
      "COMMON$FILES",
      "DIFF_VIEWER$COMMITS",
      "COMMON$TERMINAL",
      "COMMON$BROWSER",
      // Local planning is now supported, so the Planner tab is a default tab.
      "COMMON$PLANNER",
    ];
    for (const tab of expectedTabs) {
      expect(screen.getByText(tab)).toBeInTheDocument();
    }

    expect(screen.queryByText("FILES$DIFF_VIEW")).not.toBeInTheDocument();
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
    // Planner is first in the tab order, so it becomes active when files unpins.
    expect(storeState.selectedTab).toBe("planner");

    const storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${CONVERSATION_ID}`)!,
    );
    expect(storedState.unpinnedTabs).toContain("files");
    expect(storedState.selectedTab).toBe("planner");
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

  describe("portal placement", () => {
    it("anchors below and to the left of the trigger when the menu fits", () => {
      const portal = renderAnchoredMenu({ top: 20, bottom: 44, left: 300 });

      expect(portal?.style.top).toBe("52px");
      expect(portal?.style.left).toBe("300px");
      expect(portal?.style.bottom).toBe("");
    });

    it("keeps the menu inside the viewport when the trigger hugs the right edge", () => {
      // An embedded canvas can sit hard against the viewport edge; left-aligning
      // on the trigger there would clip the labels and the pin controls.
      const portal = renderAnchoredMenu({ top: 20, bottom: 44, left: 1000 });

      // 1024 (jsdom viewport) - 220 (menu) - 8 (margin)
      expect(portal?.style.left).toBe("796px");
    });

    it("flips above the trigger when the menu would clip at the viewport bottom", () => {
      const portal = renderAnchoredMenu({ top: 700, bottom: 724, left: 300 });

      // 768 (jsdom viewport) - 700 (trigger top) + 8 (gap)
      expect(portal?.style.bottom).toBe("76px");
      expect(portal?.style.top).toBe("");
    });
  });
});
