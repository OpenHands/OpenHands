import { describe, expect, it } from "vitest";
import { SETTINGS_QUERY_KEYS } from "./query-keys";

describe("SETTINGS_QUERY_KEYS", () => {
  it("returns the canonical root settings key", () => {
    expect(SETTINGS_QUERY_KEYS.all).toEqual(["settings"]);
  });

  it("builds scoped settings keys", () => {
    expect(SETTINGS_QUERY_KEYS.byScope("personal", null)).toEqual([
      "settings",
      "personal",
      null,
    ]);
    expect(SETTINGS_QUERY_KEYS.byScope("org", "org-123")).toEqual([
      "settings",
      "org",
      "org-123",
    ]);
  });

  it("builds the canonical personal settings key", () => {
    expect(SETTINGS_QUERY_KEYS.personal("org-123")).toEqual([
      "settings",
      "personal",
      "org-123",
    ]);
    expect(SETTINGS_QUERY_KEYS.personal(undefined)).toEqual(
      SETTINGS_QUERY_KEYS.byScope("personal", undefined),
    );
  });
});
