import { describe, expect, it } from "vitest";
import { SETTINGS_QUERY_KEYS } from "./query-keys";

describe("SETTINGS_QUERY_KEYS", () => {
  it("returns the canonical root settings key", () => {
    expect(SETTINGS_QUERY_KEYS.all).toEqual(["settings"]);
  });

  it("builds scoped settings keys", () => {
    expect(SETTINGS_QUERY_KEYS.byScope("personal")).toEqual([
      "settings",
      "personal",
    ]);
  });

  it("builds the canonical personal settings key", () => {
    expect(SETTINGS_QUERY_KEYS.personal()).toEqual(["settings", "personal"]);
    expect(SETTINGS_QUERY_KEYS.personal()).toEqual(
      SETTINGS_QUERY_KEYS.byScope("personal"),
    );
  });
});
