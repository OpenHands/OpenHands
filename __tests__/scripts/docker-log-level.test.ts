// @vitest-environment node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);
const blockStart = "# >>> backend-log-level";
const blockEnd = "# <<< backend-log-level";

function logLevelBlock(): string {
  const start = entrypoint.indexOf(blockStart);
  const end = entrypoint.indexOf(blockEnd);
  if (start === -1 || end === -1) {
    throw new Error("Docker backend-log-level markers are missing");
  }
  return entrypoint.slice(start, end);
}

/** Resolve LOG_LEVEL / AUTOMATION_LOG_LEVEL exactly as the entrypoint does. */
function resolveLogLevels(env: Record<string, string>): {
  LOG_LEVEL?: string;
  AUTOMATION_LOG_LEVEL?: string;
} {
  const script = [
    "set -uo pipefail",
    logLevelBlock(),
    'printf "LOG_LEVEL=%s\\n" "${LOG_LEVEL:-}"',
    'printf "AUTOMATION_LOG_LEVEL=%s\\n" "${AUTOMATION_LOG_LEVEL:-}"',
  ].join("\n");

  const result = spawnSync("bash", ["-c", script], {
    env: { PATH: process.env.PATH ?? "", ...env },
    encoding: "utf-8",
  });

  if (result.status !== 0) {
    throw new Error(`bash failed: ${result.stderr}`);
  }

  const parsed: Record<string, string> = {};
  for (const line of result.stdout.trim().split("\n")) {
    const [key, value] = line.split("=");
    parsed[key] = value;
  }
  return parsed;
}

/** Narrow a level exactly as the entrypoint does, via the marked block. */
function uvicornLogLevel(level: string): string {
  const script = [
    "set -uo pipefail",
    logLevelBlock(),
    'uvicorn_log_level "$1"',
  ].join("\n");

  const result = spawnSync("bash", ["-c", script, "bash", level], {
    env: { PATH: process.env.PATH ?? "" },
    encoding: "utf-8",
  });

  if (result.status !== 0) {
    throw new Error(`bash failed: ${result.stderr}`);
  }
  return result.stdout;
}

describe("Docker backend log level", () => {
  it("quiets both backends by default", () => {
    const levels = resolveLogLevels({});

    expect(levels.LOG_LEVEL).toBe("WARNING");
    expect(levels.AUTOMATION_LOG_LEVEL).toBe("WARNING");
  });

  it("honors an explicit LOG_LEVEL and applies it to the automation service", () => {
    const levels = resolveLogLevels({ LOG_LEVEL: "DEBUG" });

    expect(levels.LOG_LEVEL).toBe("DEBUG");
    expect(levels.AUTOMATION_LOG_LEVEL).toBe("DEBUG");
  });

  it("lets AUTOMATION_LOG_LEVEL diverge from LOG_LEVEL", () => {
    const levels = resolveLogLevels({
      LOG_LEVEL: "ERROR",
      AUTOMATION_LOG_LEVEL: "INFO",
    });

    expect(levels.LOG_LEVEL).toBe("ERROR");
    expect(levels.AUTOMATION_LOG_LEVEL).toBe("INFO");
  });

  it("passes the resolved level to every uvicorn invocation", () => {
    // The automation service's --log-level governs the startup banner and
    // per-request access lines, which LOG_LEVEL alone does not cover. The
    // invocation must route the level through uvicorn_log_level, which both
    // lowercases it and narrows it to a name Uvicorn accepts.
    const code = entrypoint
      .split("\n")
      .filter((line) => !line.trimStart().startsWith("#"))
      .join("\n");

    // Each invocation is a backslash-continued command ending in `&`.
    const invocations = code.match(
      /uvicorn openhands\.automation\.app:app[\s\S]*?&\n/g,
    );

    expect(invocations).toHaveLength(2);
    for (const invocation of invocations ?? []) {
      expect(invocation).toContain("--log-level");
      expect(invocation).toContain('uvicorn_log_level "${LOG_LEVEL}"');
    }
  });

  // Uvicorn validates --log-level with a click.Choice and exits 2 on anything
  // outside its set; the entrypoint runs without `set -e`, so a rejected value
  // would leave Canvas shipped with the automation service unavailable.
  describe("narrows --log-level to a value Uvicorn accepts", () => {
    const UVICORN_ACCEPTS = [
      "critical",
      "error",
      "warning",
      "info",
      "debug",
      "trace",
    ];

    it("keeps every level Uvicorn already accepts", () => {
      for (const level of UVICORN_ACCEPTS) {
        expect(uvicornLogLevel(level)).toBe(level);
        expect(uvicornLogLevel(level.toUpperCase())).toBe(level);
      }
    });

    it("maps the Python aliases Uvicorn rejects", () => {
      expect(uvicornLogLevel("WARN")).toBe("warning");
      expect(uvicornLogLevel("FATAL")).toBe("critical");
    });

    it("maps numeric levels onto the nearest Uvicorn name", () => {
      expect(uvicornLogLevel("50")).toBe("critical");
      expect(uvicornLogLevel("40")).toBe("error");
      expect(uvicornLogLevel("30")).toBe("warning");
      expect(uvicornLogLevel("20")).toBe("info");
      expect(uvicornLogLevel("10")).toBe("debug");
      expect(uvicornLogLevel("5")).toBe("trace");
    });

    it("falls back to Uvicorn's own default for unrepresentable values", () => {
      expect(uvicornLogLevel("NOTSET")).toBe("info");
      expect(uvicornLogLevel("0")).toBe("info");
      expect(uvicornLogLevel("")).toBe("info");
      expect(uvicornLogLevel("bogus")).toBe("info");
    });

    it("never returns a value outside Uvicorn's accepted set", () => {
      // Includes a value too large for `[` to compare as a signed integer,
      // which would otherwise error out and emit nothing.
      const inputs = [
        "WARN",
        "FATAL",
        "NOTSET",
        "30",
        "030",
        "0",
        "",
        "bogus",
        "999999999999999999999",
      ];
      for (const input of inputs) {
        expect(UVICORN_ACCEPTS).toContain(uvicornLogLevel(input));
      }
    });
  });
});
