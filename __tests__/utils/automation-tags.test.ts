import { describe, expect, it } from "vitest";
import {
  getAutomationTagError,
  normalizeAutomationTag,
} from "#/utils/automation-tags";

describe("normalizeAutomationTag", () => {
  it("trims, lowercases, and joins words with hyphens", () => {
    expect(normalizeAutomationTag("  Software \t Factory ")).toBe(
      "software-factory",
    );
  });
});

describe("getAutomationTagError", () => {
  const tenTags = Array.from({ length: 10 }, (_, i) => `tag-${i}`);

  it.each<[string, readonly string[], string, string | null]>([
    ["a new valid tag", ["ci"], "software-factory", null],
    ["a tag already on the list, even when full", tenTags, "tag-3", null],
    ["a tag with characters outside the slug set", [], "q3_release", "invalid"],
    ["a tag with a doubled hyphen", [], "software--factory", "invalid"],
    ["a tag longer than fifty characters", [], "a".repeat(51), "too-long"],
    ["an eleventh tag", tenTags, "one-more", "too-many"],
  ])("returns the right verdict for %s", (_case, tags, tag, expected) => {
    expect(getAutomationTagError(tags, tag)).toBe(expected);
  });
});
