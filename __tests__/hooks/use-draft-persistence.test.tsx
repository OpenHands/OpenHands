import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import {
  HOME_PROMPT_DRAFT_KEY,
  useDraftPersistence,
} from "#/hooks/chat/use-draft-persistence";
import * as conversationLocalStorage from "#/utils/conversation-local-storage";
import * as chatInputUtils from "#/components/features/chat/utils/chat-input.utils";

// Mock the entire module
vi.mock("#/utils/conversation-local-storage", () => ({
  useConversationLocalStorageState: vi.fn(),
  getConversationState: vi.fn(),
  setConversationState: vi.fn(),
}));

// Mock the chat-input utilities. The hook uses both `getTextContent`
// (to read the current contentEditable text) and `focusContentEditableAtEnd`
// (to place the caret at the end of the input after restoring a draft).
// Both need to be mocked here so the hook doesn't crash on
// `focusContentEditableAtEnd is not a function`.
vi.mock("#/components/features/chat/utils/chat-input.utils", () => ({
  getTextContent: vi.fn((el: HTMLDivElement | null) => el?.textContent || ""),
  focusContentEditableAtEnd: vi.fn(),
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
    sessionStorage.clear();

    mockSetDraftMessage = vi.fn<(message: string | null) => void>();

    // Default mock for useConversationLocalStorageState
    vi.mocked(
      conversationLocalStorage.useConversationLocalStorageState,
    ).mockReturnValue({
      state: {
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      },
      setSelectedTab: vi.fn(),
      setUnpinnedTabs: vi.fn(),
      setConversationMode: vi.fn(),
      setDraftMessage: mockSetDraftMessage,
      setFilesTabDiffView: vi.fn(),
      setFilesTabContentViewMode: vi.fn(),
    });

    // Default mock for getConversationState
    vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
      selectedTab: "files",
      unpinnedTabs: [],
      conversationMode: "code",
      subConversationTaskId: null,
      draftMessage: null,
      filesTabDiffView: null,
      filesTabContentViewMode: "rich",
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  describe("draft restoration on mount", () => {
    it("restores draft from localStorage when mounting with existing draft", () => {
      // Arrange
      const conversationId = "conv-restore-1";
      const savedDraft = "Previously saved draft message";
      const chatInputRef = createMockChatInputRef();

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: savedDraft,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
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
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: savedDraft,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
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
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });

      // Act
      renderHook(() => useDraftPersistence(conversationId, chatInputRef));

      // Assert - content should be cleared since there's no draft
      expect(chatInputRef.current?.textContent).toBe("");
      expect(chatInputUtils.focusContentEditableAtEnd).not.toHaveBeenCalled();
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

      vi.mocked(
        conversationLocalStorage.useConversationLocalStorageState,
      ).mockReturnValue({
        state: {
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: existingDraft,
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        },
        setSelectedTab: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
        setFilesTabDiffView: vi.fn(),
        setFilesTabContentViewMode: vi.fn(),
      });

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: existingDraft,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
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

    it("saves new draft text when no persisted draft exists", () => {
      const chatInputRef = createMockChatInputRef("Stryker was here!");
      const { result } = renderHook(() =>
        useDraftPersistence("conv-new-draft", chatInputRef),
      );
      chatInputRef.current!.textContent = "Stryker was here!";

      act(() => {
        result.current.saveDraft();
        vi.advanceTimersByTime(500);
      });

      expect(mockSetDraftMessage).toHaveBeenCalledWith("Stryker was here!");
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
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "Draft from conv A",
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        })
        .mockReturnValue({
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: null,
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
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
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: null,
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        })
        .mockReturnValue({
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: draftForConvB,
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
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

    it("switches from the home page to a real conversation safely", () => {
      const chatInputRef = createMockChatInputRef();
      const { rerender } = renderHook(
        ({ conversationId }: { conversationId: string | undefined }) =>
          useDraftPersistence(conversationId, chatInputRef),
        {
          initialProps: {
            conversationId: undefined as string | undefined,
          },
        },
      );

      expect(() => rerender({ conversationId: "conv-real" })).not.toThrow();
    });

    it("uses the current conversation setter after rerendering", () => {
      const setDraftForA = vi.fn();
      const setDraftForB = vi.fn();
      vi.mocked(
        conversationLocalStorage.useConversationLocalStorageState,
      ).mockImplementation((conversationId) => ({
        state: {
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: null,
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        },
        setSelectedTab: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage:
          conversationId === "conv-A" ? setDraftForA : setDraftForB,
        setFilesTabDiffView: vi.fn(),
        setFilesTabContentViewMode: vi.fn(),
      }));
      const chatInputRef = createMockChatInputRef();
      const { result, rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );

      rerender({ conversationId: "conv-B" });
      chatInputRef.current.textContent = "draft for B";
      act(() => {
        result.current.saveDraft();
        vi.advanceTimersByTime(500);
      });

      expect(setDraftForA).not.toHaveBeenCalled();
      expect(setDraftForB).toHaveBeenCalledWith("draft for B");
    });

    it("clears the current conversation after leaving the home page", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "home prompt");
      const chatInputRef = createMockChatInputRef();
      const { result, rerender } = renderHook(
        ({ conversationId }: { conversationId: string | undefined }) =>
          useDraftPersistence(conversationId, chatInputRef),
        {
          initialProps: {
            conversationId: undefined as string | undefined,
          },
        },
      );

      rerender({ conversationId: "conv-real" });
      act(() => result.current.clearDraft());

      expect(mockSetDraftMessage).toHaveBeenCalledWith(null);
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe("home prompt");
    });
  });

  describe("task ID to real conversation ID transition", () => {
    it("transfers draft from task ID to real conversation ID during transition", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef(
        "  Draft typed during init  ",
      );

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });

      const { result, rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "task-abc-123" } },
      );

      // Simulate user typing during task initialization
      chatInputRef.current!.textContent = "  Draft typed during init  ";

      // Act - transition to real conversation ID
      rerender({ conversationId: "conv-real-123" });

      // Assert - draft should be saved to the new real conversation ID
      expect(
        conversationLocalStorage.setConversationState,
      ).toHaveBeenCalledWith("conv-real-123", {
        draftMessage: "Draft typed during init",
      });

      // And the draft should remain visible in the input
      expect(chatInputRef.current?.textContent).toBe(
        "  Draft typed during init  ",
      );
      expect(result.current.isRestored).toBe(true);
    });

    it("does not transfer empty draft during task-to-real transition", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
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
      expect(
        conversationLocalStorage.setConversationState,
      ).not.toHaveBeenCalled();
    });

    it("does not transfer draft for non-task ID transitions", () => {
      // Arrange
      const chatInputRef = createMockChatInputRef("Some draft");

      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
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
      expect(
        conversationLocalStorage.setConversationState,
      ).not.toHaveBeenCalled();
    });
  });

  describe("hasDraft and isRestored state", () => {
    it("returns hasDraft true when draft exists in hook state", () => {
      // Arrange
      const conversationId = "conv-has-draft";
      const chatInputRef = createMockChatInputRef();

      vi.mocked(
        conversationLocalStorage.useConversationLocalStorageState,
      ).mockReturnValue({
        state: {
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "Existing draft",
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        },
        setSelectedTab: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
        setFilesTabDiffView: vi.fn(),
        setFilesTabContentViewMode: vi.fn(),
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
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: "Draft to restore",
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });

      // Act
      const { result } = renderHook(() =>
        useDraftPersistence(conversationId, chatInputRef),
      );

      // Assert
      expect(result.current.isRestored).toBe(true);
    });
  });

  describe("home-page session draft", () => {
    it("restores and preserves a saved home prompt", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "  saved home prompt  ");
      const chatInputRef = createMockChatInputRef();

      const { result, unmount } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      expect(
        conversationLocalStorage.useConversationLocalStorageState,
      ).toHaveBeenCalledWith("");
      expect(chatInputRef.current.textContent).toBe("  saved home prompt  ");
      expect(result.current.isRestored).toBe(true);

      unmount();
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe(
        "  saved home prompt  ",
      );
    });

    it("does not overwrite existing home input with a stored prompt", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "stored prompt");
      const chatInputRef = createMockChatInputRef("current prompt");

      const { unmount } = renderHook(() =>
        useDraftPersistence(null, chatInputRef),
      );

      expect(chatInputRef.current.textContent).toBe("current prompt");
      unmount();
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
    });

    it("restores safely when the browser has no Selection object", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "saved prompt");
      vi.spyOn(window, "getSelection").mockReturnValue(null);
      const chatInputRef = createMockChatInputRef();

      const { result, unmount } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      expect(chatInputRef.current.textContent).toBe("saved prompt");
      expect(result.current.isRestored).toBe(true);
      unmount();
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe(
        "saved prompt",
      );
    });

    it("restores over whitespace-only home input and places the caret at the end", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "saved prompt");
      const chatInputRef = createMockChatInputRef("   ");
      document.body.appendChild(chatInputRef.current);

      renderHook(() => useDraftPersistence(undefined, chatInputRef));

      expect(chatInputRef.current.textContent).toBe("saved prompt");
      const selection = window.getSelection();
      expect(selection?.rangeCount).toBe(1);
      const caret = selection!.getRangeAt(0);
      expect(caret.collapsed).toBe(true);
      expect(caret.startOffset).toBe(chatInputRef.current.childNodes.length);
      chatInputRef.current.remove();
    });

    it("finishes restoration when session storage is unavailable", () => {
      vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });
      const chatInputRef = createMockChatInputRef();

      const { result } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      expect(chatInputRef.current.textContent).toBe("");
      expect(result.current.isRestored).toBe(true);
    });

    it("writes non-empty input synchronously and removes an empty input", () => {
      const chatInputRef = createMockChatInputRef("  new home prompt  ");
      const { result } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      act(() => result.current.saveDraft());
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe(
        "new home prompt",
      );

      chatInputRef.current.textContent = "   ";
      act(() => result.current.saveDraft());
      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
    });

    it("does nothing when saving without a mounted input", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "existing prompt");
      const chatInputRef = { current: null };
      const { result } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      act(() => result.current.saveDraft());

      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe(
        "existing prompt",
      );
      expect(result.current.isRestored).toBe(false);
    });

    it("tolerates storage failures while saving and clearing", () => {
      const chatInputRef = createMockChatInputRef("home prompt");
      const { result } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );
      vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });

      expect(() => act(() => result.current.saveDraft())).not.toThrow();

      vi.restoreAllMocks();
      vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });
      expect(() => act(() => result.current.clearDraft())).not.toThrow();
    });

    it("clears the persisted home prompt", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "home prompt");
      const chatInputRef = createMockChatInputRef("home prompt");
      const { result } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );

      act(() => result.current.clearDraft());

      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
    });

    it("does not resurrect a prompt deliberately removed before unmount", () => {
      const chatInputRef = createMockChatInputRef("home prompt");
      const { result, unmount } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );
      act(() => result.current.saveDraft());
      sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);

      unmount();

      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
    });

    it("removes an empty tracked prompt during unmount cleanup", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "stale prompt");
      const chatInputRef = createMockChatInputRef(" ");
      const { result, unmount } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );
      chatInputRef.current.textContent = " ";
      act(() => result.current.saveDraft());

      unmount();

      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
    });

    it("tolerates storage failure during unmount cleanup", () => {
      const chatInputRef = createMockChatInputRef("home prompt");
      const { result, unmount } = renderHook(() =>
        useDraftPersistence(undefined, chatInputRef),
      );
      act(() => result.current.saveDraft());
      vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });

      expect(unmount).not.toThrow();
    });
  });

  describe("debounce race handling", () => {
    it("initializes safely when a conversation input is not mounted", () => {
      const chatInputRef = { current: null };

      const { result } = renderHook(() =>
        useDraftPersistence("conv-without-input", chatInputRef),
      );

      expect(result.current.isRestored).toBe(false);
      expect(
        conversationLocalStorage.getConversationState,
      ).not.toHaveBeenCalled();
    });

    it("ignores a stale timer callback after the conversation changes", () => {
      const chatInputRef = createMockChatInputRef();
      const { result, rerender } = renderHook(
        ({ conversationId }) =>
          useDraftPersistence(conversationId, chatInputRef),
        { initialProps: { conversationId: "conv-A" } },
      );
      chatInputRef.current.textContent = "draft for A";
      act(() => result.current.saveDraft());
      vi.spyOn(globalThis, "clearTimeout").mockImplementation(() => undefined);

      rerender({ conversationId: "conv-B" });
      chatInputRef.current.textContent = "draft for B";
      act(() => vi.advanceTimersByTime(500));

      expect(mockSetDraftMessage).not.toHaveBeenCalled();
    });

    it("does not save when the input disappears before debounce completes", () => {
      vi.mocked(
        conversationLocalStorage.useConversationLocalStorageState,
      ).mockReturnValue({
        state: {
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "existing draft",
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        },
        setSelectedTab: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
        setFilesTabDiffView: vi.fn(),
        setFilesTabContentViewMode: vi.fn(),
      });
      const chatInputRef: { current: HTMLDivElement | null } =
        createMockChatInputRef("draft");
      const { result } = renderHook(() =>
        useDraftPersistence("conv-disappearing", chatInputRef),
      );
      chatInputRef.current!.textContent = "new draft";
      act(() => result.current.saveDraft());
      chatInputRef.current = null;

      act(() => vi.advanceTimersByTime(500));

      expect(mockSetDraftMessage).not.toHaveBeenCalled();
    });

    it("saves an emptied conversation draft as null", () => {
      vi.mocked(
        conversationLocalStorage.useConversationLocalStorageState,
      ).mockReturnValue({
        state: {
          selectedTab: "files",
          unpinnedTabs: [],
          conversationMode: "code",
          subConversationTaskId: null,
          draftMessage: "old draft",
          filesTabDiffView: null,
          filesTabContentViewMode: "rich",
        },
        setSelectedTab: vi.fn(),
        setUnpinnedTabs: vi.fn(),
        setConversationMode: vi.fn(),
        setDraftMessage: mockSetDraftMessage,
        setFilesTabDiffView: vi.fn(),
        setFilesTabContentViewMode: vi.fn(),
      });
      vi.mocked(conversationLocalStorage.getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
        subConversationTaskId: null,
        draftMessage: "old draft",
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });
      const chatInputRef = createMockChatInputRef();
      const { result } = renderHook(() =>
        useDraftPersistence("conv-empty", chatInputRef),
      );
      chatInputRef.current.textContent = "   ";

      act(() => {
        result.current.saveDraft();
        vi.advanceTimersByTime(500);
      });

      expect(mockSetDraftMessage).toHaveBeenCalledWith(null);
    });
  });

  describe("cleanup on unmount", () => {
    it("does not change the home prompt when unmounting a conversation", () => {
      sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "home prompt");
      const chatInputRef = createMockChatInputRef();

      const { unmount } = renderHook(() =>
        useDraftPersistence("conv-unmount", chatInputRef),
      );
      unmount();

      expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe("home prompt");
    });

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

  it("keeps the home prompt storage key stable across module reloads", async () => {
    vi.resetModules();
    const freshModule = await import("#/hooks/chat/use-draft-persistence");

    expect(freshModule.HOME_PROMPT_DRAFT_KEY).toBe("oh:home-prompt-draft");
  });
});
