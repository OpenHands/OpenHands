import { describe, expect, it } from "vitest";

import {
  getProcessTreeSpawnOptions,
  isProcessRunning,
  resolveWindowsCommand,
} from "../../scripts/dev-process-utils.mjs";

describe("dev process utils", () => {
  it("treats a signaled but unexited process as still running", () => {
    expect(
      isProcessRunning({
        exitCode: null,
        signalCode: null,
        killed: true,
      }),
    ).toBe(true);
  });

  it("treats exited or signaled processes as stopped", () => {
    expect(isProcessRunning({ exitCode: 0, signalCode: null })).toBe(false);
    expect(isProcessRunning({ exitCode: null, signalCode: "SIGTERM" })).toBe(
      false,
    );
  });

  it("sets detached mode according to the platform for process-group cleanup", () => {
    expect(getProcessTreeSpawnOptions({ cwd: "/tmp" })).toMatchObject({
      cwd: "/tmp",
      detached: process.platform !== "win32",
    });
  });
});

describe("resolveWindowsCommand", () => {
  const lookup = (cmd: string) => `C:\\Users\\me\\.local\\bin\\${cmd}.exe`;

  it("returns the command unchanged on non-Windows platforms", () => {
    expect(resolveWindowsCommand("uvx", "linux", lookup)).toBe("uvx");
    expect(resolveWindowsCommand("uvx", "darwin", lookup)).toBe("uvx");
  });

  it("resolves a bare command to its absolute path on Windows so it can spawn without a shell", () => {
    // Spawning uvx directly (not via cmd.exe) keeps arguments such as
    // `--with agent-client-protocol<0.11` literal, instead of the `<` being
    // parsed as input redirection and failing with "The system cannot find the
    // file specified."
    expect(resolveWindowsCommand("uvx", "win32", lookup)).toBe(
      "C:\\Users\\me\\.local\\bin\\uvx.exe",
    );
  });

  it("leaves an already-resolved path untouched on Windows", () => {
    const absolute = "C:\\Windows\\System32\\cmd.exe";
    expect(resolveWindowsCommand(absolute, "win32", lookup)).toBe(absolute);
  });

  it("falls back to the original command when the lookup fails on Windows", () => {
    expect(resolveWindowsCommand("uvx", "win32", () => null)).toBe("uvx");
  });
});
