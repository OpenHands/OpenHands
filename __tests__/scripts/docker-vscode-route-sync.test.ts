// @vitest-environment node
//
// Drift-detection for the Docker install path's editor route.
//
// The VSCode button advertises a URL built by agent-server from
// OH_VSCODE_BASE_PATH, and that URL only resolves because the static server
// carries a route for the same prefix to the same port. Those two facts live in
// separate files (docker/entrypoint.sh, config/defaults.json via the Dockerfile's
// generated defaults.env), so nothing but a test stops them drifting apart and
// leaving a button that points at the canvas shell instead of the editor.
//
// The npm launcher's equivalent wiring is covered in dev-with-automation.test.ts
// against the real functions; this file asserts the shell/Docker half, which has
// no importable surface.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const defaults = JSON.parse(read("config/defaults.json")) as {
  ports: { vscode: number; proxy: number };
  paths: { vscodeBasePath: string };
};
const entrypoint = read("docker/entrypoint.sh");
const dockerfile = read("docker/Dockerfile");

// Both static-server invocations (the normal one and the --auth-required
// public-mode one started when PUBLIC_MODE_PORT is set) must carry the route;
// the public-mode server is what the auth-mode E2E suite drives.
function staticServerInvocations(): string[] {
  return entrypoint
    .split("node /opt/agent-canvas/static-server.mjs")
    .slice(1)
    .map((chunk) => chunk.split("\nSTATIC_PID")[0].split("\n  PIDS")[0]);
}

describe("docker editor route", () => {
  it("centralizes the base path and port in defaults.json", () => {
    expect(defaults.paths.vscodeBasePath).toBe("/vscode");
    expect(defaults.paths.vscodeBasePath.startsWith("/")).toBe(true);
    expect(Number.isInteger(defaults.ports.vscode)).toBe(true);
  });

  it("exports both values from defaults.json into the generated defaults.env", () => {
    // The container has no jq/python, so the Dockerfile bakes defaults.json
    // into a shell-sourceable env file. A value missing here silently falls
    // back to the hardcoded default in entrypoint.sh.
    expect(dockerfile).toContain(
      "'CONFIG_VSCODE_BASE_PATH=' + c.paths.vscodeBasePath",
    );
    expect(dockerfile).toContain("'CONFIG_VSCODE_PORT=' + c.ports.vscode");
  });

  it("passes the base path and port through to agent-server", () => {
    // agent-server launches openvscode-server with --server-base-path from
    // OH_VSCODE_BASE_PATH and includes the prefix in /api/vscode/url.
    expect(entrypoint).toMatch(
      /export OH_VSCODE_BASE_PATH="\$\{OH_VSCODE_BASE_PATH:-\$\{VSCODE_BASE_PATH\}\}"/,
    );
    expect(entrypoint).toMatch(
      /export OH_VSCODE_PORT="\$\{OH_VSCODE_PORT:-\$\{VSCODE_PORT\}\}"/,
    );
    // Defaults resolve from defaults.env, then a literal fallback.
    expect(entrypoint).toContain(
      'VSCODE_BASE_PATH="${VSCODE_BASE_PATH:-${CONFIG_VSCODE_BASE_PATH:-/vscode}}"',
    );
    expect(entrypoint).toContain(
      'VSCODE_PORT="${VSCODE_PORT:-${CONFIG_VSCODE_PORT:-8001}}"',
    );
  });

  it("registers the editor route on every static-server instance", () => {
    const invocations = staticServerInvocations();
    // Normal + public-mode. If this count changes, the new invocation needs
    // the route too.
    expect(invocations).toHaveLength(2);

    for (const invocation of invocations) {
      expect(invocation).toContain(
        '--route "${VSCODE_BASE_PATH}=http://127.0.0.1:${VSCODE_PORT}"',
      );
    }
  });

  it("routes the editor to its own port, not the agent-server", () => {
    // The editor is a separate process. Pointing the prefix at the
    // agent-server port would 404 the workbench.
    for (const invocation of staticServerInvocations()) {
      expect(invocation).not.toContain(
        '--route "${VSCODE_BASE_PATH}=http://127.0.0.1:${AGENT_SERVER_PORT}"',
      );
    }
    expect(defaults.ports.vscode).not.toBe(defaults.ports.proxy);
  });

  it("does not publish the editor port", () => {
    // The single-origin shape is the point: the editor is reachable only
    // through the proxy port's path prefix, so it inherits the canvas's
    // auth/ingress posture instead of needing a second exposed port.
    expect(dockerfile).not.toMatch(
      new RegExp(`^\\s*EXPOSE\\s+${defaults.ports.vscode}\\b`, "m"),
    );
  });
});
