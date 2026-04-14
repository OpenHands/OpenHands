import { describe, expect, it } from "vitest";
import { normalizeStringArray } from "#/utils/normalize-string-array";

describe("normalizeStringArray", () => {
  it("returns empty array for non-arrays", () => {
    expect(normalizeStringArray(undefined)).toEqual([]);
    expect(normalizeStringArray(null)).toEqual([]);
    expect(normalizeStringArray({})).toEqual([]);
    expect(normalizeStringArray("x")).toEqual([]);
  });

  it("keeps only string entries", () => {
    expect(
      normalizeStringArray(["a", 1, null, "b", {}, undefined, "c"]),
    ).toEqual(["a", "b", "c"]);
  });
});
