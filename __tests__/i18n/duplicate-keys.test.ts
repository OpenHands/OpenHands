import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";

// Keys are collected from the raw file text: JSON.parse keeps only the last
// occurrence of a duplicate key, so a duplicate is invisible once parsed.
const KEY_DEFINITION_REGEX = /"((?:[^"\\]|\\.)+)"\s*:\s*\{/g;

function findDuplicateKeys(rawJson: string): Map<string, number> {
  const occurrences = new Map<string, number>();
  for (const [, key] of rawJson.matchAll(KEY_DEFINITION_REGEX)) {
    occurrences.set(key, (occurrences.get(key) ?? 0) + 1);
  }
  return new Map([...occurrences].filter(([, count]) => count > 1));
}

describe("translation.json", () => {
  it("detects duplicate keys that JSON.parse would silently drop", () => {
    const rawJson = `{
  "COMMON$SAVE": { "en": "Save" },
  "COMMON$CANCEL":{ "en": "Cancel" },
  "COMMON$SAVE": { "en": "Store" }
}`;

    expect(Object.keys(JSON.parse(rawJson))).toHaveLength(2);
    expect(findDuplicateKeys(rawJson)).toEqual(new Map([["COMMON$SAVE", 2]]));
  });

  it("should not have duplicate translation keys", () => {
    const translationPath = path.join(
      __dirname,
      "../../src/i18n/translation.json",
    );
    const translationContent = fs.readFileSync(translationPath, "utf-8");

    const duplicates = [...findDuplicateKeys(translationContent)].map(
      ([key, count]) => `"${key}" appears ${count} times`,
    );

    expect(duplicates).toEqual([]);
  });
});
