import { beforeEach, describe, expect, it } from "vitest";
import {
  CUSTOM_SKILLS_STORAGE_KEY,
  CUSTOM_SKILLS_UPDATED_EVENT,
  deleteCustomSkill,
  formatSkillMarkdown,
  getCustomSkill,
  loadCustomSkills,
  parseSkillFrontmatter,
  saveCustomSkill,
} from "#/utils/custom-skills-storage";

describe("custom-skills-storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe("parseSkillFrontmatter", () => {
    it("parses valid YAML frontmatter and extracts body", () => {
      const raw = `---
name: my-skill
description: Does something cool
triggers:
  - test
  - demo
---

# Title
Body content here.`;

      const result = parseSkillFrontmatter(raw);
      expect(result.name).toBe("my-skill");
      expect(result.description).toBe("Does something cool");
      expect(result.triggers).toEqual(["test", "demo"]);
      expect(result.body).toBe("# Title\nBody content here.");
    });

    it("parses string triggers (comma or space separated)", () => {
      const raw = `---
name: slug-name
triggers: trigger1, trigger2
---
body`;
      const result = parseSkillFrontmatter(raw);
      expect(result.name).toBe("slug-name");
      expect(result.triggers).toEqual(["trigger1", "trigger2"]);
      expect(result.body).toBe("body");
    });

    it("handles content without frontmatter", () => {
      const raw = "# Just Markdown\nNo frontmatter here.";
      const result = parseSkillFrontmatter(raw);
      expect(result.name).toBeUndefined();
      expect(result.description).toBeUndefined();
      expect(result.triggers).toBeUndefined();
      expect(result.body).toBe(raw);
    });

    it("handles malformed frontmatter gracefully", () => {
      const raw = `---
: invalid yaml [[[[
---
Some body`;
      const result = parseSkillFrontmatter(raw);
      expect(result.body).toBe("Some body");
      expect(result.name).toBeUndefined();
      expect(result.description).toBeUndefined();
      expect(result.triggers).toBeUndefined();
    });
  });

  describe("formatSkillMarkdown", () => {
    it("formats markdown with frontmatter and body", () => {
      const formatted = formatSkillMarkdown({
        name: "my-skill",
        description: "description here",
        triggers: ["t1", "t2"],
        content: "# Body content",
      });

      expect(formatted).toContain("name: my-skill");
      expect(formatted).toContain("description: description here");
      expect(formatted).toContain("- t1");
      expect(formatted).toContain("- t2");
      expect(formatted).toContain("# Body content");
    });

    it("strips pre-existing frontmatter before formatting", () => {
      const existing = `---
name: old-name
description: old description
---
# Original Body`;

      const formatted = formatSkillMarkdown({
        name: "new-name",
        description: "new description",
        triggers: ["t3"],
        content: existing,
      });

      expect(formatted).toContain("name: new-name");
      expect(formatted).not.toContain("name: old-name");
      expect(formatted).toContain("# Original Body");
    });
  });

  describe("saveCustomSkill and loadCustomSkills", () => {
    it("saves a valid custom skill and retrieves it", () => {
      const saved = saveCustomSkill({
        name: "test-skill",
        description: "A test custom skill",
        triggers: ["test"],
        content: "# Test Skill\nInstructions for test.",
      });

      expect(saved.name).toBe("test-skill");
      expect(saved.source).toBe("custom");
      expect(saved.type).toBe("knowledge");
      expect(saved.triggers).toEqual(["test"]);

      const list = loadCustomSkills();
      expect(list).toHaveLength(1);
      expect(list[0]?.name).toBe("test-skill");
      expect(list[0]?.source).toBe("custom");
    });

    it("updates existing custom skill with same name without duplicate", () => {
      saveCustomSkill({
        name: "test-skill",
        description: "First version",
        triggers: ["test"],
        content: "Content v1",
      });

      saveCustomSkill({
        name: "test-skill",
        description: "Second version",
        triggers: ["test2"],
        content: "Content v2",
      });

      const list = loadCustomSkills();
      expect(list).toHaveLength(1);
      expect(list[0]?.description).toBe("Second version");
      expect(list[0]?.triggers).toEqual(["test2"]);
      expect(list[0]?.content).toContain("Content v2");
    });

    it("throws error for invalid slug name", () => {
      expect(() =>
        saveCustomSkill({
          name: "Invalid Name With Spaces",
          content: "content",
        }),
      ).toThrow();

      expect(() =>
        saveCustomSkill({
          name: "",
          content: "content",
        }),
      ).toThrow();
    });

    it("handles corrupted localStorage gracefully", () => {
      localStorage.setItem(CUSTOM_SKILLS_STORAGE_KEY, "invalid JSON text {");
      expect(loadCustomSkills()).toEqual([]);
    });

    it("dispatches CUSTOM_SKILLS_UPDATED_EVENT on save", () => {
      let dispatched = false;
      window.addEventListener(
        CUSTOM_SKILLS_UPDATED_EVENT,
        () => {
          dispatched = true;
        },
        { once: true },
      );

      saveCustomSkill({
        name: "event-skill",
        content: "event content",
      });

      expect(dispatched).toBe(true);
    });
  });

  describe("deleteCustomSkill", () => {
    it("deletes a custom skill by name", () => {
      saveCustomSkill({
        name: "to-delete",
        content: "content",
      });

      expect(loadCustomSkills()).toHaveLength(1);

      deleteCustomSkill("to-delete");
      expect(loadCustomSkills()).toHaveLength(0);
    });

    it("dispatches CUSTOM_SKILLS_UPDATED_EVENT on delete", () => {
      saveCustomSkill({
        name: "to-delete-event",
        content: "content",
      });

      let dispatched = false;
      window.addEventListener(
        CUSTOM_SKILLS_UPDATED_EVENT,
        () => {
          dispatched = true;
        },
        { once: true },
      );

      deleteCustomSkill("to-delete-event");
      expect(dispatched).toBe(true);
    });
  });

  describe("getCustomSkill", () => {
    it("returns null when skill is not found", () => {
      expect(getCustomSkill("nonexistent")).toBeNull();
    });

    it("returns skill info when found", () => {
      saveCustomSkill({
        name: "my-existing-skill",
        content: "hello world",
      });

      const found = getCustomSkill("my-existing-skill");
      expect(found).not.toBeNull();
      expect(found?.name).toBe("my-existing-skill");
    });
  });
});
