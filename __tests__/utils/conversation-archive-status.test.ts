import { describe, expect, it } from "vitest";
import {
  isArchivedSandboxStatus,
  isConversationArchived,
} from "#/utils/conversation-archive-status";

describe("conversation-archive-status", () => {
  describe("isArchivedSandboxStatus", () => {
    it("returns true for MISSING sandbox status", () => {
      expect(isArchivedSandboxStatus("MISSING")).toBe(true);
    });

    it("returns false for ERROR sandbox status", () => {
      expect(isArchivedSandboxStatus("ERROR")).toBe(false);
    });

    it("returns false for other sandbox statuses or empty values", () => {
      expect(isArchivedSandboxStatus("RUNNING")).toBe(false);
      expect(isArchivedSandboxStatus("PAUSED")).toBe(false);
      expect(isArchivedSandboxStatus(null)).toBe(false);
      expect(isArchivedSandboxStatus(undefined)).toBe(false);
    });
  });

  describe("isConversationArchived", () => {
    it("returns true when sandbox status is MISSING regardless of explicit flag", () => {
      expect(isConversationArchived("MISSING", false)).toBe(true);
      expect(isConversationArchived("MISSING", true)).toBe(true);
    });

    it("returns true when explicitly archived regardless of sandbox status", () => {
      expect(isConversationArchived("RUNNING", true)).toBe(true);
      expect(isConversationArchived(null, true)).toBe(true);
    });

    it("returns false for ERROR sandbox status when not explicitly archived", () => {
      expect(isConversationArchived("ERROR", false)).toBe(false);
    });

    it("returns false for active sandboxes when not explicitly archived", () => {
      expect(isConversationArchived("RUNNING", false)).toBe(false);
      expect(isConversationArchived(null, false)).toBe(false);
    });
  });
});
