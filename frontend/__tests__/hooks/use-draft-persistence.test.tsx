import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useDraftPersistence } from "#/hooks/chat/use-draft-persistence";
import * as conversationLocalStorage from "#/utils/conversation-local-storage";

// Mock the entire module
vi.mock("#/utils/conversation-local-storage", () => ({
  useConversationLocalStorageState: vi.fn(),
  getConversationState: vi.fn(),
  setConversationState: vi.fn(),
}));

// Mock the getTextContent utility
vi.mock("#/components/features/chat/utils/chat-input.utils", () => ({
  getTextContent: vi.fn((el: HTMLDivElement | null) => el?.textContent || ""),
}));

describe("useDraftPersistence", () => {
  let mockSetDraftMessage: (message: string | null) => void;

  // Create a mock ref to contentEditable div
  const createMockChatInputRef = (initialContent = "") => {
    const div = document.createElement("div");
    div.setAttribute("contenteditable", "true");
    div.textContent = initialContent;
    return { current: div };
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    localStorage.clear();

    mockSetDraftMessage = vi.fn<(message: string | null) => void>();

    // Default mock for useConversationLocalStorageState
    vi.mocked(conversationLocalStorage.useConversationLocalStorageState).mockReturnValue({
      state: {
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
      },
      setSelectedTab: vi.fn(),
      setRightPanelShown: vi.fn(),
      setUnpinnedTabs: vi.fn(),
      setConversationMode: vi.fn(),
      setDraftMessage: mockSetDraftMessage,
    });

    // Default mock for getConversationState
    vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
      selectedTab: "editor",
      rightPanelShown: true,
      unpinnedTabs: [],
      conversationMode: "code",
      subConversationTaskId: null,
      draftMessage: null,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe("draft restoration on mount", () => {
    it("restores draft from localStorage when mounting with existing draft", () => {
      // Arrange
      const conversationId = "conv-restore-1";
      const savedDraft = "Previously saved draft message";
      const chatInputRef = createMockChatInputRef();

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: savedDraft,
      });

      // Act
      renderHook(() => useDraftPersistence(conversationId, chatInputRef));

      // Assert - draft should be restored to the DOM element
      expect(chatInputRef.current?.textContent).toBe(savedDraft);
    });

    it("clears input on mount then restores draft if exists", () => {
      // Arrange
      const conversationId = "conv-restore-2";
      const existingContent = "Stale content from previous conversation";
      const savedDraft = "Saved draft";
      const chatInputRef = createMockChatInputRef(existingContent);

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: savedDraft,
      });

      // Act
      renderHook(() => useDraftPersistence(conversationId, chatInputRef));

      // Assert - input cleared then draft restored
      expect(chatInputRef.current?.textContent).toBe(savedDraft);
    });

    it("clears input when no draft exists for conversation", () => {
      // Arrange
      const conversationId = "conv-no-draft";
      const chatInputRef = createMockChatInputRef("Some stale content");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
      });

      // Act
      renderHook(() => useDraftPersistence(conversationId, chatInputRef));

      // Assert - content should be cleared since there's no draft
      expect(chatInputRef.current?.textContent).toBe("");
    });
  });

  describe("debounced saving", () => {
    it("saves draft after debounce period", () => {
      // Arrange
      const conversationId = "conv-debounce-1";
      const chatInputRef = createMockChatInputRef();

      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Act - simulate user typing
      chatInputRef.current!.textContent = "New draft content";
      act(() => {
        result.current.saveDraft();
      });

      // Assert - should not save immediately
      expect(mockSetDraftMessage).not.toHaveBeenCalled();

      // Fast forward past debounce period (500ms)
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - should save after debounce
      expect(mockSetDraftMessage).toHaveBeenCalledWith("New draft content");
    });

    it("cancels pending save when new input arrives before debounce", () => {
      // Arrange
      const conversationId = "conv-debounce-2";
      const chatInputRef = createMockChatInputRef();

      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Act - first input
      chatInputRef.current!.textContent = "First";
      act(() => {
        result.current.saveDraft();
      });

      // Wait 200ms (less than debounce)
      act(() => {
        vi.advanceTimersByTime(200);
      });

      // Second input before debounce completes
      chatInputRef.current!.textContent = "First Second";
      act(() => {
        result.current.saveDraft();
      });

      // Complete the second debounce
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - should only save the final value once
      expect(mockSetDraftMessage).toHaveBeenCalledTimes(1);
      expect(mockSetDraftMessage).toHaveBeenCalledWith("First Second");
    });

    it("does not save if content matches existing draft", () => {
      // Arrange
      const conversationId = "conv-no-change";
      const existingDraft = "Existing draft";
      const chatInputRef = createMockChatInputRef(existingDraft);

      vi.mocked(conversationLocalStorage.useConversationLocalStorageState).mockReturnValue({
        state: {
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: existingDraft,
        },
        setSelectedTab: vi.fn(),
        setRightPanelShown: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
      });

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: existingDraft,
      });

      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Act - try to save same content
      act(() => {
        result.current.saveDraft();
      });

      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - should not save since content is the same
      expect(mockSetDraftMessage).not.toHaveBeenCalled();
    });
  });

  describe("clearDraft", () => {
    it("clears the draft from localStorage", () => {
      // Arrange
      const conversationId = "conv-clear-1";
      const chatInputRef = createMockChatInputRef("Some content");

      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Act
      act(() => {
        result.current.clearDraft();
      });

      // Assert
      expect(mockSetDraftMessage).toHaveBeenCalledWith(null);
    });

    it("cancels any pending debounced save when clearing", () => {
      // Arrange
      const conversationId = "conv-clear-2";
      const chatInputRef = createMockChatInputRef();

      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Start a save
      chatInputRef.current!.textContent = "Pending draft";
      act(() => {
        result.current.saveDraft();
      });

      // Clear before debounce completes
      act(() => {
        vi.advanceTimersByTime(200);
        result.current.clearDraft();
      });

      // Complete the original debounce period
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - only the clear should have been called (the pending save should be cancelled)
      expect(mockSetDraftMessage).toHaveBeenCalledTimes(1);
      expect(mockSetDraftMessage).toHaveBeenCalledWith(null);
    });
  });

  describe("conversation switching", () => {
    it("clears input when switching to a new conversation without a draft", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("Draft from conv A");

      // First conversation has a draft
      vi.mocked(conversationLocalStorage.getConversationState)
        .mockReturnValueOnce({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "Draft from conv A",
        })
        .mockReturnValue({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: null,
        });

      const { rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );

      // Act - switch to conversation B
      rerender({ conversationId: "conv-B" });

      // Assert - input should be cleared (no draft for conv-B)
      expect(chatInputRef.current?.textContent).toBe("");
    });

    it("restores draft when switching to a conversation with an existing draft", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef();
      const draftForConvB = "Saved draft for conversation B";

      vi.mocked(conversationLocalStorage.getConversationState)
        .mockReturnValueOnce({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: null,
        })
        .mockReturnValue({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: draftForConvB,
        });

      const { rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );

      // Act - switch to conversation B
      rerender({ conversationId: "conv-B" });

      // Assert - draft for conv-B should be restored
      expect(chatInputRef.current?.textContent).toBe(draftForConvB);
    });

    it("cancels pending save when switching conversations", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef();

      const { result, rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );

      // Start typing in conv-A
      chatInputRef.current!.textContent = "Draft for conv-A";
      act(() => {
        result.current.saveDraft();
      });

      // Switch conversation before debounce completes
      act(() => {
        vi.advanceTimersByTime(200);
      });
      rerender({ conversationId: "conv-B" });

      // Complete the debounce period
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - the save should NOT have happened because conversation changed
      expect(mockSetDraftMessage).not.toHaveBeenCalled();
    });
  });

  describe("task ID to real conversation ID transition", () => {
    it("transfers draft from task ID to real conversation ID during transition", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("Draft typed during init");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
      });

      const { rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "task-abc-123" } },
      );

      // Simulate user typing during task initialization
      chatInputRef.current!.textContent = "Draft typed during init";

      // Act - transition to real conversation ID
      rerender({ conversationId: "conv-real-123" });

      // Assert - draft should be saved to the new real conversation ID
      expect(conversationLocalStorage.setConversationState).toHaveBeenCalledWith(
        "conv-real-123",
        { draftMessage: "Draft typed during init" },
      );

      // And the draft should remain visible in the input
      expect(chatInputRef.current?.textContent).toBe("Draft typed during init");
    });

    it("does not transfer empty draft during task-to-real transition", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
      });

      const { rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "task-abc-123" } },
      );

      // Act - transition to real conversation ID with empty input
      rerender({ conversationId: "conv-real-123" });

      // Assert - no draft should be saved (input is cleared, checked by hook)
      // The setConversationState should not be called with draftMessage
      expect(conversationLocalStorage.setConversationState).not.toHaveBeenCalled();
    });

    it("does not transfer draft for non-task ID transitions", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("Some draft");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
      });

      const { rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );

      // Act - normal conversation switch (not task-to-real)
      rerender({ conversationId: "conv-B" });

      // Assert - should not use setConversationState directly
      // (the normal path uses setDraftMessage from the hook)
      expect(conversationLocalStorage.setConversationState).not.toHaveBeenCalled();
    });
  });

  describe("hasDraft and isRestored state", () => {
    it("returns hasDraft true when draft exists in hook state", () => {
      // Arrange
      const conversationId = "conv-has-draft";
      const chatInputRef = createMockChatInputRef();

      vi.mocked(conversationLocalStorage.useConversationLocalStorageState).mockReturnValue({
        state: {
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "Existing draft",
        },
        setSelectedTab: vi.fn(),
        setRightPanelShown: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
      });

      // Act
      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Assert
      expect(result.current.hasDraft).toBe(true);
    });

    it("returns hasDraft false when no draft exists", () => {
      // Arrange
      const conversationId = "conv-no-draft";
      const chatInputRef = createMockChatInputRef();

      // Act
      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Assert
      expect(result.current.hasDraft).toBe(false);
    });

    it("sets isRestored to true after restoration completes", () => {
      // Arrange
      const conversationId = "conv-restored";
      const chatInputRef = createMockChatInputRef();

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: "Draft to restore",
      });

      // Act
      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Assert
      expect(result.current.isRestored).toBe(true);
    });
  });

  describe("cleanup on unmount", () => {
    it("clears pending timeout on unmount", () => {
      // Arrange
      const conversationId = "conv-unmount";
      const chatInputRef = createMockChatInputRef();

      const { result, unmount } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Start a save
      chatInputRef.current!.textContent = "Draft";
      act(() => {
        result.current.saveDraft();
      });

      // Unmount before debounce completes
      unmount();

      // Complete the debounce period
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Assert - save should not have been called after unmount
      expect(mockSetDraftMessage).not.toHaveBeenCalled();
    });
  });
});
