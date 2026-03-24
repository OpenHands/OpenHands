import { describe, it, expect, beforeEach } from "vitest";
import {
  clearConversationLocalStorage,
  getConversationState,
  isTaskConversationId,
  setConversationState,
  LOCAL_STORAGE_KEYS,
} from "#/utils/conversation-local-storage";

describe("conversation localStorage utilities", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe("isTaskConversationId", () => {
    it("returns true for IDs starting with task-", () => {
      expect(isTaskConversationId("task-abc-123")).toBe(true);
      expect(isTaskConversationId("task-")).toBe(true);
    });

    it("returns false for normal conversation IDs", () => {
      expect(isTaskConversationId("conv-123")).toBe(false);
      expect(isTaskConversationId("abc")).toBe(false);
    });
  });

  describe("getConversationState", () => {
    it("returns default state including conversationMode for task IDs without reading localStorage", () => {
      const state = getConversationState("task-uuid-123");

      expect(state.conversationMode).toBe("code");
      expect(state.selectedTab).toBe("editor");
      expect(state.rightPanelShown).toBe(true);
      expect(
        localStorage.getItem(
          `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-task-uuid-123`,
        ),
      ).toBeNull();
    });

    it("returns merged state from localStorage for real conversation ID including conversationMode", () => {
      const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-conv-1`;
      localStorage.setItem(
        key,
        JSON.stringify({ conversationMode: "plan", selectedTab: "terminal" }),
      );

      const state = getConversationState("conv-1");

      expect(state.conversationMode).toBe("plan");
      expect(state.selectedTab).toBe("terminal");
      expect(state.rightPanelShown).toBe(true);
    });

    it("returns default state when key is missing or invalid", () => {
      expect(getConversationState("conv-missing").conversationMode).toBe(
        "code",
      );

      const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-conv-bad`;
      localStorage.setItem(key, "not json");
      expect(getConversationState("conv-bad").conversationMode).toBe("code");
    });
  });

  describe("setConversationState", () => {
    it("does not persist when conversationId is a task ID", () => {
      setConversationState("task-xyz", { conversationMode: "plan" });

      expect(
        localStorage.getItem(
          `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-task-xyz`,
        ),
      ).toBeNull();
    });

    it("persists conversationMode for real conversation ID and getConversationState returns it", () => {
      setConversationState("conv-2", { conversationMode: "plan" });

      const state = getConversationState("conv-2");
      expect(state.conversationMode).toBe("plan");
    });
  });

  describe("clearConversationLocalStorage", () => {
    it("removes the consolidated conversation-state localStorage entry", () => {
      const conversationId = "conv-123";

      // Set up the consolidated key
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
        }),
      );

      clearConversationLocalStorage(conversationId);

      expect(localStorage.getItem(consolidatedKey)).toBeNull();
    });

    it("does not throw if conversation keys do not exist", () => {
      expect(() => {
        clearConversationLocalStorage("non-existent-id");
      }).not.toThrow();
    });
  });

  describe("getConversationState", () => {
    it("returns default state with subConversationTaskId as null when no state exists", () => {
      const conversationId = "conv-123";
      const state = getConversationState(conversationId);

      expect(state.subConversationTaskId).toBeNull();
      expect(state.selectedTab).toBe("editor");
      expect(state.rightPanelShown).toBe(true);
      expect(state.unpinnedTabs).toEqual([]);
    });

    it("retrieves subConversationTaskId from localStorage when it exists", () => {
      const conversationId = "conv-123";
      const taskId = "task-uuid-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "editor",
          rightPanelShown: true,
          unpinnedTabs: [],
          subConversationTaskId: taskId,
        }),
      );

      const state = getConversationState(conversationId);

      expect(state.subConversationTaskId).toBe(taskId);
    });

    it("merges stored state with defaults when partial state exists", () => {
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          subConversationTaskId: "task-123",
        }),
      );

      const state = getConversationState(conversationId);

      expect(state.subConversationTaskId).toBe("task-123");
      expect(state.selectedTab).toBe("editor");
      expect(state.rightPanelShown).toBe(true);
      expect(state.unpinnedTabs).toEqual([]);
    });
  });

  describe("setConversationState", () => {
    it("persists subConversationTaskId to localStorage", () => {
      const conversationId = "conv-123";
      const taskId = "task-uuid-456";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      setConversationState(conversationId, {
        subConversationTaskId: taskId,
      });

      const stored = localStorage.getItem(consolidatedKey);
      expect(stored).not.toBeNull();

      const parsed = JSON.parse(stored!);
      expect(parsed.subConversationTaskId).toBe(taskId);
    });

    it("merges subConversationTaskId with existing state", () => {
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      // Set initial state
      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "changes",
          rightPanelShown: false,
          unpinnedTabs: ["tab-1"],
          subConversationTaskId: "old-task-id",
        }),
      );

      // Update only subConversationTaskId
      setConversationState(conversationId, {
        subConversationTaskId: "new-task-id",
      });

      const stored = localStorage.getItem(consolidatedKey);
      const parsed = JSON.parse(stored!);

      expect(parsed.subConversationTaskId).toBe("new-task-id");
      expect(parsed.selectedTab).toBe("changes");
      expect(parsed.rightPanelShown).toBe(false);
      expect(parsed.unpinnedTabs).toEqual(["tab-1"]);
    });

    it("clears subConversationTaskId when set to null", () => {
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      // Set initial state with task ID
      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          subConversationTaskId: "task-123",
        }),
      );

      // Clear the task ID
      setConversationState(conversationId, {
        subConversationTaskId: null,
      });

      const stored = localStorage.getItem(consolidatedKey);
      const parsed = JSON.parse(stored!);

      expect(parsed.subConversationTaskId).toBeNull();
    });
  });

  describe("draftMessage persistence", () => {
    describe("getConversationState", () => {
      it("returns default draftMessage as null when no state exists", () => {
        // Arrange
        const conversationId = "conv-draft-1";

        // Act
        const state = getConversationState(conversationId);

        // Assert
        expect(state.draftMessage).toBeNull();
      });

      it("retrieves draftMessage from localStorage when it exists", () => {
        // Arrange
        const conversationId = "conv-draft-2";
        const draftText = "This is my saved draft message";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

        localStorage.setItem(
          consolidatedKey,
          JSON.stringify({
            draftMessage: draftText,
          }),
        );

        // Act
        const state = getConversationState(conversationId);

        // Assert
        expect(state.draftMessage).toBe(draftText);
      });

      it("returns null draftMessage for task conversation IDs (not persisted)", () => {
        // Arrange
        const taskId = "task-uuid-123";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${taskId}`;

        // Even if somehow there's data in localStorage for a task ID
        localStorage.setItem(
          consolidatedKey,
          JSON.stringify({
            draftMessage: "Should not be returned",
          }),
        );

        // Act
        const state = getConversationState(taskId);

        // Assert - should return default state, not the stored value
        expect(state.draftMessage).toBeNull();
      });
    });

    describe("setConversationState", () => {
      it("persists draftMessage to localStorage", () => {
        // Arrange
        const conversationId = "conv-draft-3";
        const draftText = "New draft message to save";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

        // Act
        setConversationState(conversationId, {
          draftMessage: draftText,
        });

        // Assert
        const stored = localStorage.getItem(consolidatedKey);
        expect(stored).not.toBeNull();
        const parsed = JSON.parse(stored!);
        expect(parsed.draftMessage).toBe(draftText);
      });

      it("does not persist draftMessage for task conversation IDs", () => {
        // Arrange
        const taskId = "task-draft-xyz";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${taskId}`;

        // Act
        setConversationState(taskId, {
          draftMessage: "Draft for task ID",
        });

        // Assert - nothing should be stored
        expect(localStorage.getItem(consolidatedKey)).toBeNull();
      });

      it("merges draftMessage with existing state without overwriting other fields", () => {
        // Arrange
        const conversationId = "conv-draft-4";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

        localStorage.setItem(
          consolidatedKey,
          JSON.stringify({
            selectedTab: "terminal",
            rightPanelShown: false,
            unpinnedTabs: ["tab-1", "tab-2"],
            conversationMode: "plan",
            subConversationTaskId: "task-123",
          }),
        );

        // Act
        setConversationState(conversationId, {
          draftMessage: "Updated draft",
        });

        // Assert
        const stored = localStorage.getItem(consolidatedKey);
        const parsed = JSON.parse(stored!);

        expect(parsed.draftMessage).toBe("Updated draft");
        expect(parsed.selectedTab).toBe("terminal");
        expect(parsed.rightPanelShown).toBe(false);
        expect(parsed.unpinnedTabs).toEqual(["tab-1", "tab-2"]);
        expect(parsed.conversationMode).toBe("plan");
        expect(parsed.subConversationTaskId).toBe("task-123");
      });

      it("clears draftMessage when set to null", () => {
        // Arrange
        const conversationId = "conv-draft-5";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

        localStorage.setItem(
          consolidatedKey,
          JSON.stringify({
            draftMessage: "Existing draft",
          }),
        );

        // Act
        setConversationState(conversationId, {
          draftMessage: null,
        });

        // Assert
        const stored = localStorage.getItem(consolidatedKey);
        const parsed = JSON.parse(stored!);
        expect(parsed.draftMessage).toBeNull();
      });

      it("clears draftMessage when set to empty string (stored as empty string)", () => {
        // Arrange
        const conversationId = "conv-draft-6";
        const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

        localStorage.setItem(
          consolidatedKey,
          JSON.stringify({
            draftMessage: "Existing draft",
          }),
        );

        // Act
        setConversationState(conversationId, {
          draftMessage: "",
        });

        // Assert
        const stored = localStorage.getItem(consolidatedKey);
        const parsed = JSON.parse(stored!);
        expect(parsed.draftMessage).toBe("");
      });
    });

    describe("conversation-specific draft isolation", () => {
      it("stores drafts separately for different conversations", () => {
        // Arrange
        const convA = "conv-A";
        const convB = "conv-B";
        const draftA = "Draft for conversation A";
        const draftB = "Draft for conversation B";

        // Act
        setConversationState(convA, { draftMessage: draftA });
        setConversationState(convB, { draftMessage: draftB });

        // Assert
        const stateA = getConversationState(convA);
        const stateB = getConversationState(convB);

        expect(stateA.draftMessage).toBe(draftA);
        expect(stateB.draftMessage).toBe(draftB);
      });

      it("updating one conversation draft does not affect another", () => {
        // Arrange
        const convA = "conv-isolated-A";
        const convB = "conv-isolated-B";

        setConversationState(convA, { draftMessage: "Original draft A" });
        setConversationState(convB, { draftMessage: "Original draft B" });

        // Act - update only conversation A
        setConversationState(convA, { draftMessage: "Updated draft A" });

        // Assert - conversation B should be unchanged
        const stateA = getConversationState(convA);
        const stateB = getConversationState(convB);

        expect(stateA.draftMessage).toBe("Updated draft A");
        expect(stateB.draftMessage).toBe("Original draft B");
      });

      it("clearing one conversation draft does not affect another", () => {
        // Arrange
        const convA = "conv-clear-A";
        const convB = "conv-clear-B";

        setConversationState(convA, { draftMessage: "Draft A" });
        setConversationState(convB, { draftMessage: "Draft B" });

        // Act - clear draft for conversation A
        setConversationState(convA, { draftMessage: null });

        // Assert
        const stateA = getConversationState(convA);
        const stateB = getConversationState(convB);

        expect(stateA.draftMessage).toBeNull();
        expect(stateB.draftMessage).toBe("Draft B");
      });
    });
  });
});
