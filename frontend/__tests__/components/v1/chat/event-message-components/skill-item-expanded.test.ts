import { describe, expect, it } from "vitest";
import {
  parseSkillContent,
  styleImportantTags,
} from "#/components/v1/chat/event-message-components/skill-item-expanded";

describe("parseSkillContent", () => {
  it("should extract matchInfo from content", () => {
    const content =
      "The following information has been included based on keyword match\nIt may or may not be relevant\n\nActual body content";
    const result = parseSkillContent(content);

    expect(result.matchInfo).toBe(
      "The following information has been included based on keyword match",
    );
    expect(result.body).toBe("Actual body content");
  });

  it("should extract filePath from content", () => {
    const content = "Skill location: /path/to/skill.md\n\nBody here";
    const result = parseSkillContent(content);

    expect(result.filePath).toBe("/path/to/skill.md");
    expect(result.body).toBe("Body here");
  });

  it("should handle content with both matchInfo and filePath", () => {
    const content =
      "The following information has been included based on keyword match\nIt may or may not be relevant\nSkill location: /path/to/skill.md\n(Use this path to resolve imports)\n\nBody content";
    const result = parseSkillContent(content);

    expect(result.matchInfo).toBe(
      "The following information has been included based on keyword match",
    );
    expect(result.filePath).toBe("/path/to/skill.md");
    expect(result.body).toBe("Body content");
  });

  it("should handle content with no metadata", () => {
    const content = "Just plain body content\nWith multiple lines";
    const result = parseSkillContent(content);

    expect(result.matchInfo).toBeNull();
    expect(result.filePath).toBeNull();
    expect(result.body).toBe("Just plain body content\nWith multiple lines");
  });

  it("should handle empty content", () => {
    const result = parseSkillContent("");

    expect(result.matchInfo).toBeNull();
    expect(result.filePath).toBeNull();
    expect(result.body).toBe("");
  });

  it("should skip blank lines between metadata and body", () => {
    const content = "Skill location: /path/to/skill.md\n\n\nBody after blanks";
    const result = parseSkillContent(content);

    expect(result.filePath).toBe("/path/to/skill.md");
    expect(result.body).toBe("Body after blanks");
  });
});

describe("styleImportantTags", () => {
  it("should wrap important tag content with bold markers", () => {
    const text = "Some text <important>critical info</important> more text";
    const result = styleImportantTags(text);

    expect(result).toBe("Some text **critical info** more text");
  });

  it("should handle multiple important tags", () => {
    const text =
      "<important>first</important> and <important>second</important>";
    const result = styleImportantTags(text);

    expect(result).toBe("**first** and **second**");
  });

  it("should handle multiline content inside important tags", () => {
    const text = "<important>line1\nline2</important>";
    const result = styleImportantTags(text);

    expect(result).toBe("**line1\nline2**");
  });

  it("should be case-insensitive", () => {
    const text = "<IMPORTANT>test</IMPORTANT>";
    const result = styleImportantTags(text);

    expect(result).toBe("**test**");
  });

  it("should return text unchanged if no important tags", () => {
    const text = "No important tags here";
    const result = styleImportantTags(text);

    expect(result).toBe("No important tags here");
  });

  it("should trim whitespace inside important tags", () => {
    const text = "<important>  spaced  </important>";
    const result = styleImportantTags(text);

    expect(result).toBe("**spaced**");
  });
});
