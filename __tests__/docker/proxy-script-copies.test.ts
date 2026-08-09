// @vitest-environment node
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

/**
 * Collect relative ``./foo.mjs`` imports from a Node launcher script.
 * These must be COPY'd into the Docker image next to static-server / ingress.
 */
function localMjsImports(source: string): string[] {
  const matches = source.matchAll(/from\s+"\.\/([^"]+\.mjs)"/g);
  return [...new Set([...matches].map((m) => m[1]))].sort();
}

describe("Docker image proxy script copies", () => {
  it("COPY's every local .mjs import used by static-server and ingress", () => {
    // Arrange — the image flattens scripts into /opt/agent-canvas/*.mjs
    const dockerfile = read("docker/Dockerfile");
    const required = new Set([
      "static-server.mjs",
      "ingress.mjs",
      ...localMjsImports(read("scripts/static-server.mjs")),
      ...localMjsImports(read("scripts/ingress.mjs")),
    ]);

    // Act / Assert — each required module must appear in a COPY … scripts/ line
    for (const file of [...required].sort()) {
      expect(
        dockerfile,
        `docker/Dockerfile must COPY scripts/${file} (imported by static-server or ingress)`,
      ).toMatch(new RegExp(`COPY\\s+scripts/${file.replace(/\./g, "\\.")}\\b`));
    }
  });
});
