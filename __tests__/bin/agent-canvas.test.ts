// @vitest-environment node
import { spawn } from "node:child_process";
import { once } from "node:events";
import net from "node:net";
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

describe("agent-canvas CLI", () => {
  it("shows usage info in --help output", async () => {
    const child = spawn(process.execPath, ["bin/agent-canvas.mjs", "--help"], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    const [code] = await once(child, "exit");

    expect(code).toBe(0);
    expect(stderr).toBe("");
    expect(stdout).toContain("@openhands/agent-canvas");
    expect(stdout).toContain("USAGE:");
    expect(stdout).toContain("--frontend-only");
    expect(stdout).toContain("--backend-only");
    expect(stdout).toContain("--allow-lan-session-key");
    expect(stdout).toContain("--help");
  });

  it("reports a busy ingress port once without a stack trace", async () => {
    const occupied = net.createServer();
    await new Promise<void>((resolve) =>
      occupied.listen(0, "127.0.0.1", resolve),
    );
    const port = (occupied.address() as net.AddressInfo).port;
    const state = mkdtempSync(resolve(tmpdir(), "canvas-busy-port-"));
    try {
      // The occupied port prevents any service from starting. Stub only uvx
      // discovery so this CLI diagnostic test also runs without Python tools.
      if (process.platform === "win32") {
        writeFileSync(resolve(state, "uvx.cmd"), "@exit /b 0\r\n");
      } else {
        writeFileSync(resolve(state, "uvx"), "#!/bin/sh\nexit 0\n", {
          mode: 0o700,
        });
      }
      const child = spawn(
        process.execPath,
        ["bin/agent-canvas.mjs", "--backend-only", "--port", String(port)],
        {
          cwd: repoRoot,
          env: {
            ...process.env,
            PATH: `${state}${delimiter}${process.env.PATH ?? ""}`,
            HOME: state,
            USERPROFILE: state,
            OH_CANVAS_SAFE_STATE_DIR: state,
            OH_AGENT_SERVER_LOCAL_PATH: "",
            OH_AUTOMATION_LOCAL_PATH: "",
          },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      let output = "";
      child.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        output += chunk.toString();
      });
      const [code] = await once(child, "exit");
      expect(code).toBe(1);
      expect(
        output.match(/Cannot start: the following ports are already in use/g),
        output,
      ).toHaveLength(1);
      expect(output).not.toMatch(/\n\s+at /);
      expect(output).toContain("--port <other>");
      expect(output).toContain("PORT=<other>");
    } finally {
      await new Promise<void>((resolve) => occupied.close(() => resolve()));
      rmSync(state, { recursive: true, force: true });
    }
  });

  it("does not require build/ in --backend-only mode", async () => {
    const tempRoot = mkdtempSync(resolve(tmpdir(), "agent-canvas-bin-"));
    const stubBinDir = mkdtempSync(resolve(tmpdir(), "agent-canvas-stub-bin-"));
    const badSdkDir = mkdtempSync(resolve(tmpdir(), "agent-canvas-bad-sdk-"));
    const isWindows = process.platform === "win32";

    try {
      mkdirSync(resolve(tempRoot, "bin"));
      copyFileSync(
        resolve(repoRoot, "bin", "agent-canvas.mjs"),
        resolve(tempRoot, "bin", "agent-canvas.mjs"),
      );
      symlinkSync(
        resolve(repoRoot, "scripts"),
        resolve(tempRoot, "scripts"),
        isWindows ? "junction" : "dir",
      );
      symlinkSync(
        resolve(repoRoot, "config"),
        resolve(tempRoot, "config"),
        isWindows ? "junction" : "dir",
      );

      if (isWindows) {
        writeFileSync(resolve(stubBinDir, "uvx.cmd"), "@exit /b 0\r\n");
      } else {
        const uvxPath = resolve(stubBinDir, "uvx");
        writeFileSync(uvxPath, "#!/bin/sh\nexit 0\n");
        chmodSync(uvxPath, 0o755);
      }

      const child = spawn(
        process.execPath,
        [resolve(tempRoot, "bin", "agent-canvas.mjs"), "--backend-only"],
        {
          cwd: tempRoot,
          env: {
            PATH: `${stubBinDir}${delimiter}${process.env.PATH ?? ""}`,
            HOME: process.env.HOME ?? "",
            OH_AGENT_SERVER_LOCAL_PATH: badSdkDir,
            PORT: "19912",
            OH_CANVAS_SAFE_BACKEND_PORT: "19910",
            OH_CANVAS_SAFE_AUTOMATION_PORT: "19911",
            OH_CANVAS_SAFE_VITE_PORT: "19913",
            ...(isWindows
              ? {
                  PATHEXT: process.env.PATHEXT ?? ".CMD;.EXE;.BAT;.COM",
                  SystemRoot: process.env.SystemRoot ?? "",
                  USERPROFILE: process.env.USERPROFILE ?? "",
                }
              : {}),
          },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );

      let output = "";
      child.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        output += chunk.toString();
      });

      const [code] = await once(child, "exit");

      expect(code).toBe(1);
      expect(output).toContain(
        "OH_AGENT_SERVER_LOCAL_PATH is missing expected workspace package",
      );
      expect(output).not.toContain("No build found");
    } finally {
      rmSync(tempRoot, { recursive: true, force: true });
      rmSync(stubBinDir, { recursive: true, force: true });
      rmSync(badSdkDir, { recursive: true, force: true });
    }
  });
});
