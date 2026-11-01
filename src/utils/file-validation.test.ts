import { afterEach, describe, expect, it, vi } from "vitest";

const MB = 1024 * 1024;

/**
 * The module resolves its caps once at import time from (1) the build-time
 * env vars, (2) the window globals the static server injects into index.html,
 * and (3) a compiled-in default. In the test environment only (2) and (3)
 * are reachable, so each case below gets a fresh module via resetModules.
 */
async function loadModule(windowOverrides: Record<string, unknown> = {}) {
  vi.resetModules();
  for (const [key, value] of Object.entries(windowOverrides)) {
    (window as unknown as Record<string, unknown>)[key] = value;
  }
  return await import("./file-validation");
}

const file = (name: string, bytes: number) =>
  new File([new Uint8Array(bytes)], name);

afterEach(() => {
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_MAX_FILE_SIZE__;
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_MAX_TOTAL_SIZE__;
  vi.unstubAllGlobals();
});

describe("file-validation", () => {
  it("defaults to a 3MB per-file and 3MB total limit with no config", async () => {
    const mod = await loadModule();
    expect(mod.MAX_FILE_SIZE).toBe(3 * MB);
    expect(mod.MAX_TOTAL_SIZE).toBe(3 * MB);
  });

  it("honors the injected window globals (bytes) from the static server", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_FILE_SIZE__: 512 * 1024 * 1024,
      __AGENT_CANVAS_MAX_TOTAL_SIZE__: 10 * MB,
    });
    expect(mod.MAX_FILE_SIZE).toBe(512 * 1024 * 1024);
    expect(mod.MAX_TOTAL_SIZE).toBe(10 * MB);
  });

  it("ignores window globals that are not positive finite numbers", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_FILE_SIZE__: 0,
      __AGENT_CANVAS_MAX_TOTAL_SIZE__: "10",
    });
    expect(mod.MAX_FILE_SIZE).toBe(3 * MB);
    expect(mod.MAX_TOTAL_SIZE).toBe(3 * MB);
  });

  it("rejects a single file that exceeds the per-file limit", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_FILE_SIZE__: 5 * MB,
    });
    const result = mod.validateFiles([file("big.png", 10 * MB)]);
    expect(result.isValid).toBe(false);
    expect(result.oversizedFiles).toEqual(["big.png"]);
    expect(result.errorMessage).toContain("5MB");
  });

  it("rejects a batch whose combined size exceeds the total limit", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_TOTAL_SIZE__: 5 * MB,
    });
    const result = mod.validateFiles([
      file("a.png", Math.floor(2.9 * MB)),
      file("b.png", Math.floor(2.9 * MB)),
    ]);
    expect(result.isValid).toBe(false);
    expect(result.errorMessage).toContain("exceeding the 5MB limit");
  });

  it("accepts files within the configured limits", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_FILE_SIZE__: 5 * MB,
      __AGENT_CANVAS_MAX_TOTAL_SIZE__: 10 * MB,
    });
    expect(mod.validateFiles([file("ok.png", 4 * MB)]).isValid).toBe(true);
    expect(
      mod.validateFiles([file("a.png", 4 * MB), file("b.png", 4 * MB)]).isValid,
    ).toBe(true);
  });

  it("reports the configured limit in error messages", async () => {
    const mod = await loadModule({
      __AGENT_CANVAS_MAX_FILE_SIZE__: 7 * MB,
      __AGENT_CANVAS_MAX_TOTAL_SIZE__: 7 * MB,
    });
    const one = mod.validateFiles([file("big.png", 8 * MB)]);
    expect(one.errorMessage).toContain("exceeding 7MB");
    const two = mod.validateFiles([
      file("a.png", 4 * MB),
      file("b.png", 4 * MB),
    ]);
    expect(two.errorMessage).toContain("exceeding the 7MB limit");
  });
});
