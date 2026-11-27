// @vitest-environment node
// scripts/logger.mjs treats winston as optional: the packaged Electron
// desktop app ships without the dev node_modules tree, so the dynamic
// winston import fails there and file logging must degrade to a no-op
// instead of crashing every script that calls fileLog().
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("winston", () => {
  throw new Error("winston is not installed in the packaged app");
});
vi.mock("winston-daily-rotate-file", () => {
  throw new Error("winston-daily-rotate-file is not installed in the packaged app");
});

describe("logger without winston", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("fileLog is a safe no-op that creates no log directory", async () => {
    const stateDir = join(tmpdir(), `logger-test-${process.pid}`);
    vi.stubEnv("OH_CANVAS_SAFE_STATE_DIR", stateDir);

    const { fileLog } = await import("../../scripts/logger.mjs");

    expect(() => fileLog("info", "hello from the packaged app")).not.toThrow();
    expect(existsSync(join(stateDir, "logs"))).toBe(false);
  });
});
