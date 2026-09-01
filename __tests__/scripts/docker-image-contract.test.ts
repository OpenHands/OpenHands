// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const dockerfile = readFileSync(
  path.join(repoRoot, "docker", "Dockerfile"),
  "utf-8",
);
const workflow = readFileSync(
  path.join(repoRoot, ".github", "workflows", "docker.yml"),
  "utf-8",
);
const selfHosting = readFileSync(
  path.join(repoRoot, "docs", "SELF_HOSTING.md"),
  "utf-8",
);
const packageJson = JSON.parse(
  readFileSync(path.join(repoRoot, "package.json"), "utf-8"),
) as { scripts: Record<string, string> };

interface DockerStage {
  parent: string;
  name: string;
  body: string;
}

function parseStages(source: string): DockerStage[] {
  const matches = [...source.matchAll(/^FROM\s+(\S+)\s+AS\s+(\S+)\s*$/gim)];
  return matches.map((match, index) => ({
    parent: match[1],
    name: match[2].toLowerCase(),
    body: source.slice(
      (match.index ?? 0) + match[0].length,
      matches[index + 1]?.index ?? source.length,
    ),
  }));
}

const stages = parseStages(dockerfile);

function stage(name: string): DockerStage {
  const result = stages.find((candidate) => candidate.name === name);
  if (!result) throw new Error(`Missing Docker stage: ${name}`);
  return result;
}

describe("Docker image target contract", () => {
  it("defines shared, sandbox, control-plane, and compatibility targets", () => {
    expect(stages.map(({ name }) => name)).toEqual(
      expect.arrayContaining([
        "agent-runtime-base",
        "enterprise-sandbox",
        "dev-sandbox",
        "control-plane-runtime",
        "control-plane",
        "final",
      ]),
    );
    expect(stages.at(-1)?.name).toBe("final");
  });

  it("builds both sandbox targets directly from the shared runtime", () => {
    expect(stage("enterprise-sandbox").parent).toBe("agent-runtime-base");
    expect(stage("dev-sandbox").parent).toBe("agent-runtime-base");
  });

  it("keeps control-plane behavior out of the Enterprise sandbox target", () => {
    expect(stage("enterprise-sandbox").body).not.toMatch(
      /ENTRYPOINT|CMD|frontend|automation|entrypoint\.sh/i,
    );
  });

  it("keeps the legacy all-in-one image as the default target", () => {
    expect(stage("final").parent).toBe("control-plane-runtime");
    expect(stage("final").body).toContain(
      'ENTRYPOINT ["tini", "--", "/opt/agent-canvas/entrypoint.sh"]',
    );
  });
});

describe("Enterprise sandbox distribution contract", () => {
  it("builds and publishes only the supported Enterprise architecture", () => {
    expect(workflow).toContain("target: enterprise-sandbox");
    expect(workflow).toContain("platforms: linux/amd64");
    expect(workflow).toContain("agent-canvas-enterprise-sandbox");
    expect(workflow).toContain(
      "load: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.repo.fork }}",
    );
    expect(workflow).toContain(
      "push: ${{ github.event_name != 'pull_request' || github.event.pull_request.head.repo.fork == false }}",
    );
    expect(workflow).toContain(
      "provenance: ${{ github.event_name != 'pull_request' || github.event.pull_request.head.repo.fork == false }}",
    );
    expect(workflow).toContain(
      "sbom: ${{ github.event_name != 'pull_request' || github.event.pull_request.head.repo.fork == false }}",
    );
  });

  it("exposes symmetric local sandbox build commands", () => {
    expect(packageJson.scripts["build:docker:dev-sandbox"]).toBe(
      "node scripts/docker-build.mjs --target dev-sandbox",
    );
    expect(packageJson.scripts["build:docker:enterprise"]).toBe(
      "node scripts/docker-build.mjs --enterprise",
    );
  });

  it("documents the Enterprise Admin Console image configuration", () => {
    expect(selfHosting).toContain("OpenHands Enterprise sandbox image");
    expect(selfHosting).toContain("Sandbox Image Tag");
    expect(selfHosting).toContain("npm run build:docker:enterprise");
  });
});
