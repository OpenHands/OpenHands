// @vitest-environment node
import { describe, expect, it } from "vitest";

import { chooseWindowToFocus } from "../../../electron/lib/window-focus-selection.mjs";

function makeWindow(overrides = {}) {
  return {
    isDestroyed: () => false,
    isMinimized: () => false,
    focus: () => {},
    restore: () => {},
    ...overrides,
  };
}

describe("chooseWindowToFocus", () => {
  it("returns mainWin when it is alive", () => {
    const mainWin = makeWindow();
    const loadingWin = makeWindow();
    expect(chooseWindowToFocus(mainWin, loadingWin)).toBe(mainWin);
  });

  it("falls back to loadingWin when mainWin is destroyed", () => {
    const mainWin = makeWindow({ isDestroyed: () => true });
    const loadingWin = makeWindow();
    expect(chooseWindowToFocus(mainWin, loadingWin)).toBe(loadingWin);
  });

  it("returns null when both windows are destroyed", () => {
    const mainWin = makeWindow({ isDestroyed: () => true });
    const loadingWin = makeWindow({ isDestroyed: () => true });
    expect(chooseWindowToFocus(mainWin, loadingWin)).toBeNull();
  });

  it("returns null when both windows are absent", () => {
    expect(chooseWindowToFocus(null, null)).toBeNull();
    expect(chooseWindowToFocus(undefined, undefined)).toBeNull();
  });

  it("returns loadingWin when mainWin is null", () => {
    const loadingWin = makeWindow();
    expect(chooseWindowToFocus(null, loadingWin)).toBe(loadingWin);
  });
});
