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
  path.join(repoRoot, "docker", "entrypoint.sh"),
  "utf-8",
);

const BLOCK_START = "# >>> agent-server-topology";
const BLOCK_END = "# <<< agent-server-topology";

function topologyBlock(): string {
  const start = entrypoint.indexOf(BLOCK_START);
  const end = entrypoint.indexOf(BLOCK_END);
  if (start === -1 || end === -1) {
    throw new Error("docker/entrypoint.sh is missing the topology markers");
  }
  return entrypoint.slice(start, end);
}

function resolveTopology(env: Record<string, string> = {}) {
  const script = [
    "set -uo pipefail",
    `log_error() { printf 'ERROR: %s\\n' "$*" >&2; }`,
    'AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-18000}"',
    topologyBlock(),
    `printf '%s\\n%s\\n%s\\n' "$AGENT_CANVAS_AGENT_SERVER_MODE" "$AGENT_SERVER_PROXY_URL" "$VSCODE_HOST"`,
  ].join("\n");
  const result = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env: { PATH: process.env.PATH ?? "", ...env },
  });
  const [mode = "", proxyUrl = "", vscodeHost = ""] = result.stdout
    .trim()
    .split("\n");
  return { mode, proxyUrl, status: result.status, stderr: result.stderr, vscodeHost };
}

describe("Docker Agent Server topology", () => {
  it("preserves the embedded loopback default", () => {
    expect(resolveTopology()).toMatchObject({
      mode: "embedded",
      proxyUrl: "http://127.0.0.1:18000",
      status: 0,
      vscodeHost: "127.0.0.1",
    });
  });

  it("uses the explicit service URL in external mode", () => {
    expect(
      resolveTopology({
        AGENT_CANVAS_AGENT_SERVER_MODE: "external",
        AGENT_SERVER_URL: "http://agent-server:8000",
        VSCODE_HOST: "agent-server",
      }),
    ).toMatchObject({
      mode: "external",
      proxyUrl: "http://agent-server:8000",
      status: 0,
      vscodeHost: "agent-server",
    });
  });

  it("refuses external mode without an Agent Server URL", () => {
    const resolved = resolveTopology({
      AGENT_CANVAS_AGENT_SERVER_MODE: "external",
    });
    expect(resolved.status).not.toBe(0);
    expect(resolved.stderr).toContain("AGENT_SERVER_URL is required");
  });

  it("routes every Agent Server endpoint through the resolved URL", () => {
    for (const route of [
      "/api",
      "/server_info",
      "/sockets",
      "/alive",
      "/health",
      "/ready",
      "/docs",
      "/redoc",
      "/openapi.json",
    ]) {
      expect(entrypoint).toContain(
        `--route "${route}=\${AGENT_SERVER_PROXY_URL}"`,
      );
    }
  });

  it("uses the resolved sandbox host for the editor route", () => {
    expect(entrypoint).toContain(
      'VSCODE_ROUTE="${VSCODE_BASE_PATH}=http://${VSCODE_HOST}:${VSCODE_PORT}"',
    );
  });

  it("starts the Agent Server only in embedded mode", () => {
    expect(entrypoint).toMatch(
      /if \[ "\$AGENT_CANVAS_AGENT_SERVER_MODE" = "embedded" \]; then[\s\S]+openhands-agent-server[\s\S]+fi/,
    );
  });
});
