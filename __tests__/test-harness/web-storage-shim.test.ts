import { afterEach, describe, expect, it, vi } from "vitest";

describe("vitest Web Storage shim stability", () => {
  afterEach(() => {
    // Reproduces the order-dependent cleanup used by storage-heavy tests (e.g.
    // conversation-runtime-info.test.ts). A previous version of this suite
    // dropped setup-owned localStorage on Node 25+ here, breaking the next
    // test against the broken built-in Storage.
    vi.unstubAllGlobals();
  });

  it("keeps a working localStorage after a test that calls vi.unstubAllGlobals() in its own afterEach", () => {
    expect(typeof localStorage.setItem).toBe("function");
    localStorage.setItem("shim-driver-key", "from-first-test");
  });

  it("offers fully functional Web Storage to the test scheduled immediately after the cleanup", () => {
    // Runs after the previous test's `vi.unstubAllGlobals()`. The setup file's
    // global beforeEach must have re-established the shim, so read/write/clear
    // all work here regardless of Node version.
    localStorage.clear();

    expect(typeof localStorage.setItem).toBe("function");
    localStorage.setItem("shim-key", "shim-value");
    expect(localStorage.getItem("shim-key")).toBe("shim-value");
    localStorage.removeItem("shim-key");
    expect(localStorage.getItem("shim-key")).toBeNull();
    localStorage.setItem("shim-clear", "1");
    expect(localStorage.length).toBe(1);

    localStorage.clear();
    expect(localStorage.length).toBe(0);
  });
});
