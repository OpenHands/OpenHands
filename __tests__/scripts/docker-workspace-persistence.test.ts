// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildConversationWorkingDir,
  getAgentServerWorkingDir,
} from "#/api/agent-server-config";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const dockerfile = readFileSync(
  path.join(repoRoot, "docker/Dockerfile"),
  "utf-8",
);

/**
 * The frontend build stage, where the `VITE_*` values Vite inlines into the
 * bundle are declared.
 */
function frontendBuildStage(): string {
  const start = dockerfile.indexOf("FROM node:24-slim AS frontend-build");
  const end = dockerfile.indexOf("\nFROM ", start + 1);
  if (start === -1 || end === -1) {
    throw new Error("frontend-build stage is missing from docker/Dockerfile");
  }
  return dockerfile.slice(start, end);
}

/** The working dir the image bakes into the frontend bundle. */
function bakedWorkingDir(): string {
  const match = frontendBuildStage().match(
    /^ARG VITE_WORKING_DIR="?([^"\n]+)"?$/m,
  );
  return match?.[1] ?? "";
}

/** The directories the image declares as volumes, i.e. the persisted ones. */
function declaredVolumes(): string[] {
  const match = dockerfile.match(/^VOLUME \[(.+)\]$/m);
  if (!match) return [];
  return JSON.parse(`[${match[1]}]`) as string[];
}

/** The directories the image pre-creates for the `openhands` user. */
function persistenceDirectories(): string[] {
  const start = dockerfile.indexOf("RUN mkdir -p ");
  const end = dockerfile.indexOf("&&", start);
  if (start === -1 || end === -1) {
    throw new Error(
      "persistence directories are missing from docker/Dockerfile",
    );
  }
  return dockerfile
    .slice(start + "RUN mkdir -p ".length, end)
    .split("\\")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

/** The paths the image hands to the `openhands` group. */
function chownedPaths(): string[] {
  const match = dockerfile.match(/chown -R openhands:openhands (.+)$/m);
  if (!match) return [];
  return match[1].trim().split(/\s+/);
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Docker workspace persistence", () => {
  it("bakes a working dir into the frontend build", () => {
    const stage = frontendBuildStage();

    expect(stage).toMatch(/^ARG VITE_WORKING_DIR="[^"\n]+"$/m);
    expect(stage).toMatch(/^ENV VITE_WORKING_DIR=\$\{VITE_WORKING_DIR\}$/m);
  });

  it("bakes an absolute working dir so it is not anchored to the backend home", () => {
    // A relative base is resolved against the agent-server's home
    // (/home/openhands), which the image does not persist.
    expect(path.isAbsolute(bakedWorkingDir())).toBe(true);
  });

  it("keeps the baked working dir inside a persisted volume", () => {
    const baked = bakedWorkingDir();
    const owner = declaredVolumes().find(
      (volume) => baked === volume || baked.startsWith(`${volume}/`),
    );

    expect(owner).toBeDefined();
  });

  it("pre-creates the baked working dir for the openhands user", () => {
    const baked = bakedWorkingDir();
    const owner = declaredVolumes().find(
      (volume) => baked === volume || baked.startsWith(`${volume}/`),
    );

    expect(persistenceDirectories()).toContain(baked);
    expect(chownedPaths()).toContain(owner);
  });

  it("gives every conversation a working dir under the baked base", () => {
    const baked = bakedWorkingDir();
    vi.stubEnv("VITE_WORKING_DIR", baked);

    expect(getAgentServerWorkingDir()).toBe(baked);
    expect(
      buildConversationWorkingDir("4a8dca37-3bf0-48de-a0af-949d711c3d48"),
    ).toBe(`${baked}/4a8dca373bf048dea0af949d711c3d48`);
  });
});
