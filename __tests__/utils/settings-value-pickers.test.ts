import { describe, expect, it } from "vitest";
import {
  pickFirstBoolean,
  pickFirstNumber,
  pickNullableString,
} from "#/utils/settings-value-pickers";

describe("settings value selection", () => {
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
