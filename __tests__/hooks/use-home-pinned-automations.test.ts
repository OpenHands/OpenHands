import { describe, expect, it } from "vitest";
import {
  applyPinnedOrder,
  movePinnedId,
} from "#/hooks/use-home-pinned-automations";

describe("movePinnedId", () => {
  it("reorders an id before or after a target", () => {
    expect(movePinnedId(["a", "b", "c"], "c", "a", "before")).toEqual([
      "c",
      "a",
      "b",
    ]);
    expect(movePinnedId(["a", "b", "c"], "a", "b", "after")).toEqual([
      "b",
      "a",
      "c",
    ]);
  });
});

describe("applyPinnedOrder", () => {
  it("applies a preferred order while keeping unknown base ids", () => {
    expect(applyPinnedOrder(["a", "b", "c"], ["c", "a"])).toEqual([
      "c",
      "a",
      "b",
    ]);
  });
});
