import { afterEach, describe, expect, it, vi } from "vitest";
import { needsDesktopTitlebar } from "#/utils/desktop-shell";

describe("needsDesktopTitlebar", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("is off in the browser", () => {
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 (Macintosh)" });
    expect(needsDesktopTitlebar()).toBe(false);
  });

  it("is on for the macOS Electron shell", () => {
    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Electron/42.3.3 Safari/537.36",
    });
    expect(needsDesktopTitlebar()).toBe(true);
  });

  it("is off for the Windows Electron shell (native title bar)", () => {
    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Electron/42.3.3 Safari/537.36",
    });
    expect(needsDesktopTitlebar()).toBe(false);
  });
});
