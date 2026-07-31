import { describe, expect, it, vi, afterEach } from "vitest";
import {
  formatPrimaryModifierShortcut,
  isMacPlatform,
  matchesPrimaryModifierShortcut,
} from "#/utils/keyboard-shortcut";

describe("keyboard-shortcut", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("formats Mac shortcuts with the command glyph", () => {
    vi.stubGlobal("navigator", { platform: "MacIntel" });
    expect(formatPrimaryModifierShortcut("k")).toBe("⌘K");
    expect(isMacPlatform()).toBe(true);
  });

  it("formats non-Mac shortcuts with Ctrl+", () => {
    vi.stubGlobal("navigator", { platform: "Win32" });
    expect(formatPrimaryModifierShortcut("k")).toBe("Ctrl+K");
    expect(isMacPlatform()).toBe(false);
  });

  it("matches ⌘/Ctrl + key on any platform", () => {
    vi.stubGlobal("navigator", { platform: "MacIntel" });
    expect(
      matchesPrimaryModifierShortcut(
        new KeyboardEvent("keydown", { key: "k", metaKey: true }),
        "k",
      ),
    ).toBe(true);
    expect(
      matchesPrimaryModifierShortcut(
        new KeyboardEvent("keydown", { key: "k", ctrlKey: true }),
        "k",
      ),
    ).toBe(true);
    expect(
      matchesPrimaryModifierShortcut(
        new KeyboardEvent("keydown", { key: "k", metaKey: true, shiftKey: true }),
        "k",
      ),
    ).toBe(false);
  });
});
