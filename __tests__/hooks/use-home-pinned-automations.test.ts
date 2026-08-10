import { describe, expect, it } from "vitest";
import {
  applyPinnedOrder,
  getHomePinnedAutomationsKey,
  HOME_PINNED_AUTOMATIONS_KEY,
  movePinnedId,
} from "#/hooks/use-home-pinned-automations";

describe("getHomePinnedAutomationsKey", () => {
  it("scopes the storage key by backend and org", () => {
    expect(getHomePinnedAutomationsKey("backend-a", "org-1")).toBe(
      `${HOME_PINNED_AUTOMATIONS_KEY}:backend-a:org-1`,
    );
    expect(getHomePinnedAutomationsKey("backend-a", null)).toBe(
      `${HOME_PINNED_AUTOMATIONS_KEY}:backend-a:-`,
    );
    expect(getHomePinnedAutomationsKey("backend-a", "org-1")).not.toBe(
      getHomePinnedAutomationsKey("backend-b", "org-1"),
    );
  });
});

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
