import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useConversationStore } from "#/stores/conversation-store";

const TEST_CONVERSATION_ID = "test-conversation-id";

vi.mock("#/hooks/use-conversation-id", () => ({
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
        result.current.selectTab("editor");
      });

      // Assert: Panel should be open and tab selected
      expect(useConversationStore.getState().selectedTab).toBe("editor");
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${TEST_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.selectedTab).toBe("editor");
      expect(storedState.rightPanelShown).toBe(true);
    });

    it("should close panel when clicking the same active tab", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "editor",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Act: Click the same tab again
      act(() => {
        result.current.selectTab("editor");
      });

      // Assert: Panel should be closed
      expect(useConversationStore.getState().hasRightPanelToggled).toBe(false);

      // Verify localStorage was updated
      const storedState = JSON.parse(
        localStorage.getItem(
          `conversation-state-${TEST_CONVERSATION_ID}`,
        )!,
      );
      expect(storedState.rightPanelShown).toBe(false);
    });

    it("should switch to different tab when panel is already open", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "editor",
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
        selectedTab: "editor",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Editor tab should be active
      expect(result.current.isTabActive("editor")).toBe(true);
    });

    it("should return false when tab is selected but panel is not visible", () => {
      // Arrange: Editor tab selected but panel is closed
      useConversationStore.setState({
        selectedTab: "editor",
        isRightPanelShown: false,
        hasRightPanelToggled: false,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Editor tab should not be active
      expect(result.current.isTabActive("editor")).toBe(false);
    });

    it("should return false when different tab is selected", () => {
      // Arrange: Panel is open with editor tab selected
      useConversationStore.setState({
        selectedTab: "editor",
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
        selectedTab: "editor",
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
        selectedTab: "vscode",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Should return current selectedTab
      expect(result.current.selectedTab).toBe("vscode");
    });

    it("should return current isRightPanelShown from store", () => {
      // Arrange
      useConversationStore.setState({
        selectedTab: "editor",
        isRightPanelShown: true,
        hasRightPanelToggled: true,
      });

      const { result } = renderHook(() => useSelectConversationTab());

      // Assert: Should return current panel state
      expect(result.current.isRightPanelShown).toBe(true);
    });
  });
});
