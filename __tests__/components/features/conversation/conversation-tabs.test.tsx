import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { useConversationStore } from "#/stores/conversation-store";
import { AgentState } from "#/types/agent-state";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import {
  ACTIVE_BACKEND_STORAGE_KEY,
  BACKENDS_STORAGE_KEY,
} from "#/api/backend-registry/storage";
import type { Backend } from "#/api/backend-registry/types";

const TASK_CONVERSATION_ID = "task-ec03fb2ab8604517b24af632b058c2fd";
const REAL_CONVERSATION_ID = "conv-abc123";

let mockConversationId = TASK_CONVERSATION_ID;

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: mockConversationId }),
}));

let mockHasTaskList = false;
vi.mock("#/hooks/use-task-list", () => ({
  useTaskList: () => ({
    hasTaskList: mockHasTaskList,
    taskList: [],
  }),
}));

const mockRefetchGitChanges = vi.fn();
let mockIsFetchingGitChanges = false;
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => ({
    refetch: mockRefetchGitChanges,
    isFetching: mockIsFetchingGitChanges,
    data: [],
  }),
}));

const mockHandleBuildPlanClick = vi.fn();
vi.mock("#/hooks/use-handle-build-plan-click", () => ({
  useHandleBuildPlanClick: () => ({
    handleBuildPlanClick: mockHandleBuildPlanClick,
  }),
}));

let mockCurAgentState = AgentState.AWAITING_USER_INPUT;
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: mockCurAgentState }),
}));

vi.mock("#/hooks/query/use-unified-vscode-url", () => ({
  useUnifiedVSCodeUrl: () => ({
    data: { url: "http://localhost:8001", error: null },
    isLoading: false,
    refetch: vi.fn().mockResolvedValue({ data: { url: "http://localhost:8001" } }),
  }),
}));

const createWrapper = (conversationId: string) =>
  function ({ children }: { children: React.ReactNode }) {
    return (
      <MemoryRouter initialEntries={[`/conversations/${conversationId}`]}>
        <QueryClientProvider client={new QueryClient()}>
          <ActiveBackendProvider>{children}</ActiveBackendProvider>
        </QueryClientProvider>
      </MemoryRouter>
    );
  };

const seedConversationState = (
  conversationId: string,
  overrides: Record<string, unknown> = {},
) => {
  localStorage.setItem(
    `conversation-state-${conversationId}`,
    JSON.stringify({
      selectedTab: "files",
      unpinnedTabs: [],
      conversationMode: "code",
      subConversationTaskId: null,
      draftMessage: null,
      ...overrides,
    }),
  );
};

function seedActiveBackend(backend: Backend): void {
  localStorage.setItem(BACKENDS_STORAGE_KEY, JSON.stringify([backend]));
  localStorage.setItem(
    ACTIVE_BACKEND_STORAGE_KEY,
    JSON.stringify({ backendId: backend.id, orgId: null }),
  );
  __resetActiveStoreForTests();
}

const setActiveTabState = (tab: "files" | "planner") => {
  seedConversationState(REAL_CONVERSATION_ID, {
    selectedTab: tab,
  });
  useConversationStore.setState({
    selectedTab: tab,
    isRightPanelShown: true,
    hasRightPanelToggled: true,
  });
};

describe("ConversationTabs localStorage behavior", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetActiveStoreForTests();
    vi.resetAllMocks();
    mockRefetchGitChanges.mockReset();
    mockHandleBuildPlanClick.mockReset();
    mockConversationId = TASK_CONVERSATION_ID;
    mockHasTaskList = false;
    mockIsFetchingGitChanges = false;
    mockCurAgentState = AgentState.AWAITING_USER_INPUT;
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
      planContent: null,
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

      const changesTab = screen.getByTestId("conversation-tab-files");
      await user.click(changesTab);

      const consolidatedKey = `conversation-state-${REAL_CONVERSATION_ID}`;
      const storedState = localStorage.getItem(consolidatedKey);
      expect(storedState).not.toBeNull();

      const parsed = JSON.parse(storedState!);
      expect(parsed).toHaveProperty("selectedTab");
      expect(parsed).toHaveProperty("unpinnedTabs");
      // The right-drawer open state is session-only and must never
      // be persisted into the consolidated conversation-state blob.
      expect(parsed).not.toHaveProperty("rightPanelShown");
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

      // Assert: Panel should be open and terminal tab selected (in-memory only).
      expect(useConversationStore.getState().selectedTab).toBe("terminal");
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

      // Tab selection persists to localStorage; drawer-open state does not.
      const storedState = JSON.parse(
        localStorage.getItem(`conversation-state-${REAL_CONVERSATION_ID}`)!,
      );
      expect(storedState.selectedTab).toBe("terminal");
      expect(storedState).not.toHaveProperty("rightPanelShown");
    });

    it("should close panel when clicking the same active tab", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Act: Click the editor tab again
      const editorTab = screen.getByTestId("conversation-tab-files");
      await user.click(editorTab);

      // Assert: Panel should be closed (in-memory only).
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(false);

      // localStorage must NOT carry the drawer-open state — that's
      // session-only by design.
      const raw = localStorage.getItem(
        `conversation-state-${REAL_CONVERSATION_ID}`,
      );
      if (raw !== null) {
        expect(JSON.parse(raw)).not.toHaveProperty("rightPanelShown");
      }
    });

    it("should switch to different tab when clicking another tab while panel is open", async () => {
      mockConversationId = REAL_CONVERSATION_ID;
      const user = userEvent.setup();

      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
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
        localStorage.getItem(`conversation-state-${REAL_CONVERSATION_ID}`)!,
      );
      expect(storedState.selectedTab).toBe("browser");
    });
  });

  describe("tab action buttons", () => {
    beforeEach(() => {
      mockConversationId = REAL_CONVERSATION_ID;
    });

    it("no longer renders the refresh button in the top tab bar (it now lives inside the Files tab toolbar)", () => {
      setActiveTabState("files");

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // The old conversation-tabs refresh button used aria-label "COMMON$FILES"
      // on a top-bar <button>. Our `conversation-tab-files` button uses a
      // different DOM shape, so any <button> matching that aria-label here
      // would be the legacy refresh button.
      const buttons = Array.from(
        document.querySelectorAll('button[aria-label="COMMON$FILES"]'),
      );
      // The only remaining match should be the tab nav itself (a button with
      // data-testid conversation-tab-files), if anything. There must be no
      // standalone refresh button.
      const refreshButtons = buttons.filter(
        (b) => b.getAttribute("data-testid") !== "conversation-tab-files",
      );
      expect(refreshButtons).toHaveLength(0);
    });

    it("places the Files tab leftmost in the tab bar", () => {
      setActiveTabState("files");

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      const tabs = Array.from(
        document.querySelectorAll('[data-testid^="conversation-tab-"]'),
      );
      const testIds = tabs.map((t) => t.getAttribute("data-testid"));
      // Files must be the first tab rendered in the bar.
      expect(testIds[0]).toBe("conversation-tab-files");
    });

    it("keeps Files leftmost even when the task list tab is present", () => {
      setActiveTabState("files");
      mockHasTaskList = true;

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      const tabs = Array.from(
        document.querySelectorAll('[data-testid^="conversation-tab-"]'),
      );
      const testIds = tabs.map((t) => t.getAttribute("data-testid"));
      expect(testIds[0]).toBe("conversation-tab-files");
      // Task list should still be visible, just not first.
      expect(testIds).toContain("conversation-tab-tasklist");
    });

    it("shows an unpinned tab in the bar while it is selected", () => {
      mockConversationId = REAL_CONVERSATION_ID;
      // Planner is cloud-only, so a cloud backend is required to exercise
      // the unpinned-while-selected logic against the planner tab.
      seedActiveBackend({
        id: "cloud-test",
        name: "Cloud Test",
        host: "https://app.example.com",
        apiKey: "secret",
        kind: "cloud",
      });
      seedConversationState(REAL_CONVERSATION_ID, {
        selectedTab: "planner",
        unpinnedTabs: ["planner"],
      });
      useConversationStore.setState({
        selectedTab: "planner",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        screen.getByTestId("conversation-tab-planner"),
      ).toBeInTheDocument();
    });

    it("hides an unpinned tab from the bar once another tab is selected", () => {
      mockConversationId = REAL_CONVERSATION_ID;
      // Cloud backend so the planner tab would be eligible — it stays
      // hidden here because it is unpinned and not the selected tab.
      seedActiveBackend({
        id: "cloud-test",
        name: "Cloud Test",
        host: "https://app.example.com",
        apiKey: "secret",
        kind: "cloud",
      });
      seedConversationState(REAL_CONVERSATION_ID, {
        selectedTab: "files",
        unpinnedTabs: ["planner"],
      });
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        screen.queryByTestId("conversation-tab-planner"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("conversation-tab-files")).toBeInTheDocument();
    });

    it("does not show the build button when the planner tab is inactive", () => {
      setActiveTabState("files");
      useConversationStore.setState({
        planContent: "# Plan content",
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        screen.queryByTestId("planner-tab-build-button"),
      ).not.toBeInTheDocument();
    });

    it("shows the build button when the planner tab is active", async () => {
      setActiveTabState("planner");
      useConversationStore.setState({
        planContent: "# Plan content",
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        await screen.findByTestId("planner-tab-build-button"),
      ).toBeInTheDocument();
    });

    it("disables the build button when there is no plan content", async () => {
      setActiveTabState("planner");

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        await screen.findByTestId("planner-tab-build-button"),
      ).toBeDisabled();
    });

    it("disables the build button when the agent is running", async () => {
      mockCurAgentState = AgentState.RUNNING;
      setActiveTabState("planner");
      useConversationStore.setState({
        planContent: "# Plan content",
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        await screen.findByTestId("planner-tab-build-button"),
      ).toBeDisabled();
    });

    it("calls the build handler when the build button is clicked", async () => {
      const user = userEvent.setup();
      setActiveTabState("planner");
      useConversationStore.setState({
        planContent: "# Plan content",
      });

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      await user.click(await screen.findByTestId("planner-tab-build-button"));

      expect(mockHandleBuildPlanClick).toHaveBeenCalledTimes(1);
    });
  });

  describe("planner tab visibility by backend kind", () => {
    beforeEach(() => {
      mockConversationId = REAL_CONVERSATION_ID;
    });

    it("should hide the planner tab when the active backend is local", () => {
      // Arrange
      seedActiveBackend({
        id: "local-test",
        name: "Local Test",
        host: "http://localhost:8000",
        apiKey: "",
        kind: "local",
      });

      // Act
      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Assert
      expect(
        screen.queryByTestId("conversation-tab-planner"),
      ).not.toBeInTheDocument();
    });

    it("should show the planner tab when the active backend is cloud", () => {
      // Arrange
      seedActiveBackend({
        id: "cloud-test",
        name: "Cloud Test",
        host: "https://app.example.com",
        apiKey: "secret",
        kind: "cloud",
      });

      // Act
      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Assert
      expect(
        screen.getByTestId("conversation-tab-planner"),
      ).toBeInTheDocument();
    });
  });

  describe("vscode link visibility by backend kind", () => {
    beforeEach(() => {
      mockConversationId = REAL_CONVERSATION_ID;
    });

    it("should hide the vscode link when the active backend is local", () => {
      // Arrange
      seedActiveBackend({
        id: "local-test",
        name: "Local Test",
        host: "http://localhost:8000",
        apiKey: "",
        kind: "local",
      });

      // Act
      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Assert
      expect(
        screen.queryByTestId("drawer-vscode-link"),
      ).not.toBeInTheDocument();
    });

    it("should show the vscode link when the active backend is cloud", () => {
      // Arrange
      seedActiveBackend({
        id: "cloud-test",
        name: "Cloud Test",
        host: "https://app.example.com",
        apiKey: "secret",
        kind: "cloud",
      });

      // Act
      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      // Assert
      expect(screen.getByTestId("drawer-vscode-link")).toBeInTheDocument();
    });
  });

  describe("ellipsis context menu", () => {
    beforeEach(() => {
      mockConversationId = REAL_CONVERSATION_ID;
    });

    it("opens the context menu when the ellipsis button is clicked", async () => {
      const user = userEvent.setup();

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        screen.queryByTestId("conversation-tabs-menu-open-files"),
      ).not.toBeInTheDocument();

      await user.click(screen.getByTestId("ellipsis-button"));

      expect(
        screen.getByTestId("conversation-tabs-menu-open-files"),
      ).toBeInTheDocument();
    });
  });

  describe("tasklist tab", () => {
    beforeEach(() => {
      mockConversationId = REAL_CONVERSATION_ID;
      mockHasTaskList = true;
    });

    it("should show tasklist tab when hasTaskList is true", () => {
      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      expect(
        screen.getByTestId("conversation-tab-tasklist"),
      ).toBeInTheDocument();
    });

    it("should select tasklist tab when clicked", async () => {
      const user = userEvent.setup();

      render(<ConversationTabs />, {
        wrapper: createWrapper(REAL_CONVERSATION_ID),
      });

      const tasklistTab = screen.getByTestId("conversation-tab-tasklist");
      await user.click(tasklistTab);

      const { selectedTab, hasRightPanelToggled } =
        useConversationStore.getState();
      expect(selectedTab).toBe("tasklist");
      expect(hasRightPanelToggled).toBe(true);
    });
  });
});
