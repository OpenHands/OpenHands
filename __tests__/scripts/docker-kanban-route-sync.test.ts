// @vitest-environment node
//
// The space board UI calls /api/boards on the canvas origin. Those routes live
// in tools/kanban_api.py, not the agent-server, so the Docker static-server
// must proxy them to the sidecar or every board open 404s.
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
  ports: { kanban: number };
};
const entrypoint = read("docker/entrypoint.sh");
const dockerfile = read("docker/Dockerfile");

function staticServerInvocations(): string[] {
  return entrypoint
    .split("node /opt/agent-canvas/static-server.mjs")
    .slice(1)
    .map((chunk) => chunk.split("\nSTATIC_PID")[0].split("\n  PIDS")[0]);
}

describe("docker kanban route", () => {
  it("centralizes the sidecar port in defaults.json", () => {
    expect(defaults.ports.kanban).toBe(18004);
  });

  it("exports the port from defaults.json into the generated defaults.env", () => {
    expect(dockerfile).toContain("'CONFIG_KANBAN_PORT=' + c.ports.kanban");
  });

  it("starts the kanban sidecar before the static server", () => {
    expect(entrypoint).toContain("kanban_api.py");
    expect(entrypoint).toContain("$KANBAN_PORT");
  });

  it("proxies board routes on every static-server instance", () => {
    const invocations = staticServerInvocations();
    expect(invocations).toHaveLength(2);
    for (const invocation of invocations) {
      expect(invocation).toContain(
        '--route "/api/boards=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/columns=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/cards=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/project=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/channels=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/meetings=http://127.0.0.1:${KANBAN_PORT}"',
      );
      expect(invocation).toContain(
        '--route "/api/standards=http://127.0.0.1:${KANBAN_PORT}"',
      );
    }
  });
});
