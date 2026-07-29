import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useConversationStore } from "#/stores/conversation-store";

const TEST_CONVERSATION_ID = "test-conversation-id";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: TEST_CONVERSATION_ID }),
}));

describe("useSelectConversationTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });
  });

  describe("selectTab", () => {
    it("should open panel and select tab when panel is closed", () => {
      // Arrange: Panel is closed
      useConversationStore.setState({
        selectedTab: null,
        isRightPanelShown: false,
        hasRightPanelToggled: false,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Select a tab
      act(() => {
        result.current.selectTab("files");
      });

      // Assert: Panel should be open and tab selected (in-memory only).
      expect(useConversationStore.getState().selectedTab).toBe("files");
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

      // Tab selection is persisted; the right-drawer open state is
      // intentionally session-only and must NOT touch localStorage.
      const storedState = JSON.parse(
        localStorage.getItem(`conversation-state-${TEST_CONVERSATION_ID}`)!,
      );
      expect(storedState.selectedTab).toBe("files");
      expect(storedState).not.toHaveProperty("rightPanelShown");
    });

    it("should close panel when clicking the same active tab", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Click the same tab again
      act(() => {
        result.current.selectTab("files");
      });

      // Assert: Panel should be closed (in-memory only).
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(false);

      // The drawer-close shouldn't have written to localStorage at all
      // (session-only behavior). If anything is persisted, it's just the
      // pre-existing tab selection from earlier writes — never a
      // `rightPanelShown` field.
      const raw = localStorage.getItem(
        `conversation-state-${TEST_CONVERSATION_ID}`,
      );
      if (raw !== null) {
        expect(JSON.parse(raw)).not.toHaveProperty("rightPanelShown");
      }
    });

    it("should switch to different tab when panel is already open", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Select a different tab
      act(() => {
        result.current.selectTab("terminal");
      });

      // Assert: New tab should be selected, panel still open
      expect(useConversationStore.getState().selectedTab).toBe("terminal");
      expect(useConversationStore.getState().isRightPanelShown).toBe(true);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${TEST_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe("terminal");
    });
  });

  describe("isTabActive", () => {
    it("should return true when tab is selected and panel is visible", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Editor tab should be active
      expect(result.current.isTabActive("files")).toBe(true);
    });

    it("should return false when tab is selected but panel is not visible", () => {
      // Arrange: Editor tab selected but panel is closed
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: false,
        hasRightPanelToggled: false,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Editor tab should not be active
      expect(result.current.isTabActive("files")).toBe(false);
    });

    it("should return false when different tab is selected", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Terminal tab should not be active
      expect(result.current.isTabActive("terminal")).toBe(false);
    });
  });

  describe("onTabChange", () => {
    it("should update both Zustand store and localStorage when changing tab", () => {
      // Arrange
      useConversationStore.setState({
        selectedTab: null,
        isRightPanelShown: false,
        hasRightPanelToggled: false,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Change tab
      act(() => {
        result.current.onTabChange("browser");
      });

      // Assert: Both store and localStorage should be updated
      expect(useConversationStore.getState().selectedTab).toBe("browser");

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${TEST_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe("browser");
    });

    it("should set tab to null when passing null", () => {
      // Arrange
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Set tab to null
      act(() => {
        result.current.onTabChange(null);
      });

      // Assert: Tab should be null
      expect(useConversationStore.getState().selectedTab).toBe(null);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${TEST_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe(null);
    });
  });

  describe("returned values", () => {
    it("should return current selectedTab from store", () => {
      // Arrange
      useConversationStore.setState({
        selectedTab: "browser",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Should return current selectedTab
      expect(result.current.selectedTab).toBe("browser");
    });

    it("should return current isRightPanelShown from store", () => {
      // Arrange
      useConversationStore.setState({
        selectedTab: "files",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Should return current panel state
      expect(result.current.isRightPanelShown).toBe(true);
    });
  });
});
