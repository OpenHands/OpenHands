import { describe, expect, it } from "vitest";

import { SECURE_WEB_PREFERENCES } from "./window-security.mjs";

describe("SECURE_WEB_PREFERENCES", () => {
  it.each([
    ["nodeIntegration", false],
    ["contextIsolation", true],
    ["sandbox", true],
  ])("states %s explicitly as %s", (key, value) => {
    expect(SECURE_WEB_PREFERENCES[key]).toBe(value);
  });

  it("cannot be weakened by a caller that spreads and mutates it", () => {
    expect(Object.isFrozen(SECURE_WEB_PREFERENCES)).toBe(true);
    expect(() => {
      "use strict";
      SECURE_WEB_PREFERENCES.sandbox = false;
    }).toThrow();
    expect(SECURE_WEB_PREFERENCES.sandbox).toBe(true);
  });
});
