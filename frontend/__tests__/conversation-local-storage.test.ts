// @vitest-environment happy-dom
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  clearConversationLocalStorage,
  cleanupOrphanedConversationLocalStorage,
  LOCAL_STORAGE_KEYS,
} from "#/utils/conversation-local-storage";


describe("conversation localStorage utilities", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe("clearConversationLocalStorage", () => {
    it("removes all conversation-specific localStorage entries", () => {
      const conversationId = "conv-123";

      localStorage.setItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_SELECTED_TAB}-${conversationId}`,
        "editor",
      );
      localStorage.setItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_RIGHT_PANEL_SHOWN}-${conversationId}`,
        "true",
      );
      localStorage.setItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_UNPINNED_TABS}-${conversationId}`,
        "[]",
      );

      clearConversationLocalStorage(conversationId);

      expect(
        localStorage.getItem(
          `${LOCAL_STORAGE_KEYS.CONVERSATION_SELECTED_TAB}-${conversationId}`,
        ),
      ).toBeNull();
      expect(
        localStorage.getItem(
          `${LOCAL_STORAGE_KEYS.CONVERSATION_RIGHT_PANEL_SHOWN}-${conversationId}`,
        ),
      ).toBeNull();
      expect(
        localStorage.getItem(
          `${LOCAL_STORAGE_KEYS.CONVERSATION_UNPINNED_TABS}-${conversationId}`,
        ),
      ).toBeNull();
    });

    it("does not throw if conversation keys do not exist", () => {
      expect(() => {
        clearConversationLocalStorage("non-existent-id");
      }).not.toThrow();
    });
  });

  describe("cleanupOrphanedConversationLocalStorage", () => {
    it("removes orphaned conversation localStorage entries", () => {
      localStorage.setItem(
        "conversation-selected-tab-orphaned",
        "editor",
      );
      localStorage.setItem(
        "conversation-right-panel-shown-orphaned",
        "true",
      );

      cleanupOrphanedConversationLocalStorage(["active-id"]);

      expect(
        localStorage.getItem("conversation-selected-tab-orphaned"),
      ).toBeNull();
      expect(
        localStorage.getItem("conversation-right-panel-shown-orphaned"),
      ).toBeNull();
    });

    it("keeps localStorage entries for existing conversations", () => {
      const activeId = "active-id";

      localStorage.setItem(
        `conversation-selected-tab-${activeId}`,
        "editor",
      );

      cleanupOrphanedConversationLocalStorage([activeId]);

      expect(
        localStorage.getItem(
          `conversation-selected-tab-${activeId}`,
        ),
      ).toBe("editor");
    });

    it("does not remove unrelated localStorage keys", () => {
      localStorage.setItem("desktop-layout-panel-width", "50");

      cleanupOrphanedConversationLocalStorage([]);

      expect(
        localStorage.getItem("desktop-layout-panel-width"),
      ).toBe("50");
    });
  });
});
