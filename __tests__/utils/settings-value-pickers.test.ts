import { describe, expect, it } from "vitest";
import {
  isNonEmptyString,
  pickFirstBoolean,
  pickFirstNumber,
  pickFirstString,
  pickNullableString,
} from "#/utils/settings-value-pickers";

describe("settings value selection", () => {
  describe("non-empty strings", () => {
    it.each([
      ["a non-empty string", "configured", true],
      ["an empty string", "", false],
      ["null", null, false],
      ["a non-string with a length", { length: 1 }, false],
      ["a number", 42, false],
    ])("identifies %s", (_description, value, expected) => {
      expect(isNonEmptyString(value)).toBe(expected);
    });

    it("returns the first non-empty string and skips other values", () => {
      expect(pickFirstString(undefined, "", false, "first", "second")).toBe(
        "first",
      );
    });

    it("returns undefined when no non-empty string is available", () => {
      expect(pickFirstString(undefined, null, false, 0, "")).toBeUndefined();
      expect(pickFirstString()).toBeUndefined();
    });
  });

  describe("booleans", () => {
    it("preserves false and returns the first boolean", () => {
      expect(pickFirstBoolean("false", 0, false, true)).toBe(false);
    });

    it("returns undefined when no boolean is available", () => {
      expect(pickFirstBoolean(undefined, null, "true", 1)).toBeUndefined();
      expect(pickFirstBoolean()).toBeUndefined();
    });
  });

  describe("numbers", () => {
    it("preserves zero and returns the first number", () => {
      expect(pickFirstNumber("0", false, 0, 17)).toBe(0);
    });

    it("returns undefined when no number is available", () => {
      expect(pickFirstNumber(undefined, null, false, "1")).toBeUndefined();
      expect(pickFirstNumber()).toBeUndefined();
    });
  });

  describe("nullable strings", () => {
    it("preserves an empty string as the first explicit string value", () => {
      expect(pickNullableString(undefined, false, "", null, "later")).toBe("");
    });

    it("preserves null when it appears before a later string", () => {
      expect(pickNullableString(1, null, "later")).toBeNull();
    });

    it("returns the first string when it appears before null", () => {
      expect(pickNullableString({}, "first", null, "second")).toBe("first");
    });

    it("returns undefined when no string or null is available", () => {
      expect(
        pickNullableString(undefined, false, 42, { length: 1 }),
      ).toBeUndefined();
      expect(pickNullableString()).toBeUndefined();
    });
  });
});
