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
      expect(state.selectedTab).toBe("files");
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
    });

    it("round-trips rightPanelShown through localStorage", () => {
      const conversationId = "conv-right-panel";
      setConversationState(conversationId, {
        selectedTab: "terminal",
        rightPanelShown: true,
        unpinnedTabs: ["browser"],
      });

      const state = getConversationState(conversationId);

      expect(state.selectedTab).toBe("terminal");
      expect(state.unpinnedTabs).toEqual(["browser"]);
      expect(state.rightPanelShown).toBe(true);
    });

    it("defaults rightPanelShown to false and drops corrupt values", () => {
      expect(getConversationState("conv-right-panel-default").rightPanelShown).toBe(
        false,
      );

      const conversationId = "conv-right-panel-corrupt";
      const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
      localStorage.setItem(
        key,
        JSON.stringify({
          selectedTab: "terminal",
          rightPanelShown: "yes",
        }),
      );

      expect(getConversationState(conversationId).rightPanelShown).toBe(false);
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

      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "editor",
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
      expect(state.selectedTab).toBe("files");
      expect(state.unpinnedTabs).toEqual([]);
      expect(state.unpinnedOverviewSections).toEqual([]);
      expect(state.unpinnedOverviewGitParts).toEqual([]);
    });

    it("persists and sanitizes unpinnedOverviewSections", () => {
      const conversationId = "conv-overview-pins";
      setConversationState(conversationId, {
        unpinnedOverviewSections: ["skills", "not-a-section", "mcp", "workspace"],
      });

      const state = getConversationState(conversationId);
      // Legacy section ids (mcp/skills/secrets/…) are dropped by the allowlist.
      expect(state.unpinnedOverviewSections).toEqual(["workspace"]);
    });

    it("persists and sanitizes unpinnedOverviewGitParts", () => {
      const conversationId = "conv-overview-git-pins";
      setConversationState(conversationId, {
        unpinnedOverviewGitParts: ["branch", "not-a-part", "issues"],
      });

      const state = getConversationState(conversationId);
      // Legacy git part ids (issues) are dropped by the allowlist.
      expect(state.unpinnedOverviewGitParts).toEqual(["branch"]);
    });

    it("retrieves subConversationTaskId from localStorage when it exists", () => {
      const conversationId = "conv-123";
      const taskId = "task-uuid-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "editor",
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
      expect(state.selectedTab).toBe("files");
      expect(state.unpinnedTabs).toEqual([]);
    });

    it("falls back to the default tab when stored selectedTab is no longer valid", () => {
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      // Persisted from a previous app version where "editor" was a tab.
      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "editor",
          unpinnedTabs: [],
        }),
      );

      const state = getConversationState(conversationId);

      expect(state.selectedTab).toBe("files");
    });

    it("migrates a stored Diffs (changes) tab selection to Commits", () => {
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "changes",
          unpinnedTabs: [],
        }),
      );

      const state = getConversationState(conversationId);

      expect(state.selectedTab).toBe("commits");
    });

    it("filters obsolete tabs out of stored unpinnedTabs (editor / served / app / changes)", () => {
      // Returning users may have unpinned the now-removed Editor, Served,
      // App, or Diffs (`changes`) tabs in a previous version. Those names
      // should not survive the read — otherwise they linger forever in
      // localStorage since the UI has no way to surface them again to be
      // re-pinned.
      const conversationId = "conv-123";
      const consolidatedKey = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;

      localStorage.setItem(
        consolidatedKey,
        JSON.stringify({
          selectedTab: "files",
          unpinnedTabs: ["editor", "changes", "served", "app", "terminal"],
        }),
      );

      const state = getConversationState(conversationId);

      // Obsolete names are dropped; still-valid `terminal` stays.
      expect(state.unpinnedTabs).toEqual(["terminal"]);
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
          selectedTab: "browser",
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
      expect(parsed.selectedTab).toBe("browser");
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

  describe("filesTabDiffView preference", () => {
    it("preserves filesTabDiffView from stored blobs on read", () => {
      const conversationId = "files-diff-legacy";
      localStorage.setItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`,
        JSON.stringify({
          selectedTab: "files",
          filesTabDiffView: true,
        }),
      );

      const state = getConversationState(conversationId);
      expect(state.filesTabDiffView).toBe(true);
    });
  });

  describe("filesTabContentViewMode persistence", () => {
    // The rich/plain toggle for the file content viewer also persists
    // per conversation. Default is "rich" — verified explicitly here so
    // a careless change to the default field initializer doesn't slip
    // through unnoticed (it would flip every existing user from rich to
    // plain after deploy).

    it("defaults to 'rich' when nothing is stored", () => {
      const state = getConversationState("files-view-conv-1");
      expect(state.filesTabContentViewMode).toBe("rich");
    });

    it("round-trips 'plain' through localStorage", () => {
      const conversationId = "files-view-conv-2";
      setConversationState(conversationId, {
        filesTabContentViewMode: "plain",
      });

      expect(getConversationState(conversationId).filesTabContentViewMode).toBe(
        "plain",
      );

      const raw = localStorage.getItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`,
      );
      expect(JSON.parse(raw as string).filesTabContentViewMode).toBe("plain");
    });

    it("round-trips 'rich' through localStorage (explicit save, not default)", () => {
      const conversationId = "files-view-conv-3";
      setConversationState(conversationId, {
        filesTabContentViewMode: "rich",
      });

      expect(getConversationState(conversationId).filesTabContentViewMode).toBe(
        "rich",
      );
    });

    it("is isolated per conversation", () => {
      setConversationState("files-view-convA", {
        filesTabContentViewMode: "plain",
      });
      setConversationState("files-view-convB", {
        filesTabContentViewMode: "rich",
      });

      expect(
        getConversationState("files-view-convA").filesTabContentViewMode,
      ).toBe("plain");
      expect(
        getConversationState("files-view-convB").filesTabContentViewMode,
      ).toBe("rich");
    });

    it("falls back to the 'rich' default when localStorage holds a junk value", () => {
      // A corrupted entry (older build with a renamed mode, a hand-edited
      // value in devtools, …) must not leak through to the ViewMode-typed
      // consumer — the sanitizer drops the bad value so the merged result
      // re-applies the typed default.
      const conversationId = "files-view-corrupt";
      const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
      localStorage.setItem(
        key,
        JSON.stringify({ filesTabContentViewMode: "fancy" }),
      );

      const state = getConversationState(conversationId);
      expect(state.filesTabContentViewMode).toBe("rich");
    });
  });

  describe("files tab open-state / tree persistence", () => {
    it("defaults to an expanded tree and no open files", () => {
      const state = getConversationState("files-open-defaults");
      expect(state.filesTabTreeVisible).toBe(true);
      expect(state.filesTabOpenPaths).toEqual([]);
      expect(state.filesTabSelectedPath).toBeNull();
    });

    it("round-trips tree visibility and open tabs", () => {
      const conversationId = "files-open-roundtrip";
      setConversationState(conversationId, {
        filesTabTreeVisible: false,
        filesTabOpenPaths: ["README.md", "src/main.ts"],
        filesTabSelectedPath: "src/main.ts",
      });

      const state = getConversationState(conversationId);
      expect(state.filesTabTreeVisible).toBe(false);
      expect(state.filesTabOpenPaths).toEqual(["README.md", "src/main.ts"]);
      expect(state.filesTabSelectedPath).toBe("src/main.ts");
    });

    it("sanitizes corrupt open-state fields", () => {
      const conversationId = "files-open-corrupt";
      const key = `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`;
      localStorage.setItem(
        key,
        JSON.stringify({
          filesTabTreeVisible: "yes",
          filesTabOpenPaths: ["ok.ts", 12, "", null],
          filesTabSelectedPath: { path: "nope" },
        }),
      );

      const state = getConversationState(conversationId);
      expect(state.filesTabTreeVisible).toBe(true);
      expect(state.filesTabOpenPaths).toEqual(["ok.ts"]);
      expect(state.filesTabSelectedPath).toBeNull();
    });
  });
});
