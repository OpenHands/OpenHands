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

function browserToolsBlock(): string {
  const start = entrypoint.indexOf("# >>> docker-browser-tools");
  const end = entrypoint.indexOf("# <<< docker-browser-tools");
  if (start === -1 || end === -1) {
    throw new Error("Docker browser-tools markers are missing");
  }
  return entrypoint.slice(start, end);
}

function resolveEnableBrowser(env: Record<string, string>): string {
  const result = spawnSync(
    "bash",
    [
      "-c",
      [
        "set -uo pipefail",
        browserToolsBlock(),
        'printf "%s" "${OH_ENABLE_BROWSER:-unset}"',
      ].join("\n"),
    ],
    { encoding: "utf-8", env: { PATH: process.env.PATH ?? "", ...env } },
  );
  expect(result.status).toBe(0);
  return result.stdout;
}

describe("Docker browser-tools flag", () => {
  it("leaves the agent-server default alone unless the flag is false", () => {
    expect(resolveEnableBrowser({})).toBe("unset");
    expect(resolveEnableBrowser({ VITE_ENABLE_BROWSER_TOOLS: "true" })).toBe(
      "unset",
    );
  });

  it("turns the browser off when the flag is false", () => {
    expect(resolveEnableBrowser({ VITE_ENABLE_BROWSER_TOOLS: "false" })).toBe(
      "false",
    );
  });

  it("keeps an explicit OH_ENABLE_BROWSER", () => {
    expect(
      resolveEnableBrowser({
        VITE_ENABLE_BROWSER_TOOLS: "false",
        OH_ENABLE_BROWSER: "true",
      }),
    ).toBe("true");
  });
});
