// @vitest-environment node
//
// Drift-detection for the Docker entrypoint's shared static-server route table.
//
// The same ten agent-server/automation routes must be registered by both
// static-server instances (the normal one and the public-mode one). They now
// live in one marked block; this test asserts that single definition
// statically, and on non-Windows platforms executes the block under bash to
// confirm it produces exactly the ten routes.
import { spawnSync } from "node:child_process";
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

const entrypoint = read("docker/entrypoint.sh");

// Both static-server invocations (the normal one and the --auth-required
// public-mode one started when PUBLIC_MODE_PORT is set) must carry the shared
// routes.
function staticServerInvocations(): string[] {
  return entrypoint
    .split("node /opt/agent-canvas/static-server.mjs")
    .slice(1)
    .map((chunk) => chunk.split("\nSTATIC_PID")[0].split("\n  PIDS")[0]);
}

const BLOCK_START = "# >>> docker-route-table";
const BLOCK_END = "# <<< docker-route-table";

function routeTableBlock(): string {
  const start = entrypoint.indexOf(BLOCK_START);
  const end = entrypoint.indexOf(BLOCK_END);
  if (start === -1 || end === -1) {
    throw new Error(
      `docker/entrypoint.sh is missing the "${BLOCK_START}"/"${BLOCK_END}" markers; ` +
        "the route-table block can no longer be located, so its behavior is untested.",
    );
  }
  return entrypoint.slice(start, end);
}

const EXPECTED_ROUTES = [
  "/api/automation=http://127.0.0.1:${AUTOMATION_PORT}",
  "/api=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/server_info=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/sockets=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/alive=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/health=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/ready=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/docs=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/redoc=http://127.0.0.1:${AGENT_SERVER_PORT}",
  "/openapi.json=http://127.0.0.1:${AGENT_SERVER_PORT}",
];

function normalInvocation(invocations: string[]): string {
  return invocations.find((inv) => inv.includes('--route "$VSCODE_ROUTE"'))!;
}

function publicInvocation(invocations: string[]): string {
  return invocations.find((inv) => !inv.includes('--route "$VSCODE_ROUTE"'))!;
}

describe("docker shared static-server route table", () => {
  it("defines the route table in one marked block", () => {
    const starts = (entrypoint.match(/# >>> docker-route-table/g) ?? []).length;
    const ends = (entrypoint.match(/# <<< docker-route-table/g) ?? []).length;
    expect(starts).toBe(1);
    expect(ends).toBe(1);
  });

  it("uses the shared array on both static-server invocations", () => {
    const invocations = staticServerInvocations();
    expect(invocations).toHaveLength(2);
    invocations.forEach((invocation) => {
      expect(invocation).toContain('"${STATIC_SERVER_ROUTES[@]}"');
    });
  });

  it("the marked block defines the ten agent-server/automation routes", () => {
    const block = routeTableBlock();
    EXPECTED_ROUTES.forEach((route) => {
      expect(block).toContain(route);
    });
  });

  it("registers the shared routes on the normal static-server instance", () => {
    const invocations = staticServerInvocations();
    const normal = normalInvocation(invocations);
    expect(normal).toContain('"${STATIC_SERVER_ROUTES[@]}"');
    expect(normal).toContain('--route "$VSCODE_ROUTE"');
  });

  it("registers the shared routes on the public-mode instance", () => {
    const invocations = staticServerInvocations();
    const publicMode = publicInvocation(invocations);
    expect(publicMode).toBeDefined();
    expect(publicMode).toContain('"${STATIC_SERVER_ROUTES[@]}"');
  });

  it("keeps the editor route off the public-mode instance", () => {
    const invocations = staticServerInvocations();
    const publicMode = publicInvocation(invocations);
    expect(publicMode).not.toContain("VSCODE_ROUTE");
    expect(publicMode).not.toContain("--vscode-base-path");
  });
});

// Bash execution of the extracted block is skipped on Windows, where bash is
// not a given (mirrors docker-vscode-route-sync.test.ts).
describe.skipIf(process.platform === "win32")(
  "docker route table block execution",
  () => {
    function resolveRouteTable(env: Record<string, string> = {}): {
      status: number | null;
      stderr: string;
      routes: string[];
    } {
      const script = [
        "set -uo pipefail",
        `AUTOMATION_PORT="${env.AUTOMATION_PORT ?? "18001"}"`,
        `AGENT_SERVER_PORT="${env.AGENT_SERVER_PORT ?? "18000"}"`,
        routeTableBlock(),
        "printf '%s\\n' \"${STATIC_SERVER_ROUTES[@]}\"",
      ].join("\n");

      const res = spawnSync("bash", ["-c", script], {
        encoding: "utf-8",
        env: { PATH: process.env.PATH ?? "", ...env },
      });
      const routes = res.stdout.trim().split("\n").filter(Boolean);
      return { status: res.status, stderr: res.stderr, routes };
    }

    const RESOLVED_ROUTES = EXPECTED_ROUTES.map((r) =>
      r
        .replace("${AUTOMATION_PORT}", "18001")
        .replace("${AGENT_SERVER_PORT}", "18000"),
    );

    it("produces the ten agent-server/automation routes exactly once", () => {
      const { status, stderr, routes } = resolveRouteTable();
      expect(status).toBe(0);
      expect(routes).toEqual(RESOLVED_ROUTES);
      expect(stderr).not.toContain("unbound");
    });

    it("substitutes overridden ports", () => {
      const { status, routes } = resolveRouteTable({
        AUTOMATION_PORT: "19001",
        AGENT_SERVER_PORT: "19000",
      });
      expect(status).toBe(0);
      expect(routes).toContain("/api/automation=http://127.0.0.1:19001");
      expect(routes).toContain("/api=http://127.0.0.1:19000");
    });
  },
);
