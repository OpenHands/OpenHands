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
  path.join(repoRoot, "docker/Dockerfile"),
  "utf-8",
);
const stages = dockerfile.split(/^(?=FROM )/m);
// Match the build stage by name: its base image tag moves on its own.
const buildStage = stages.find((stage) => / AS frontend-build$/m.test(stage));

// The base dir the image bakes into the bundle; conversations get <base>/<hex>.
// A relative base is anchored to the agent-server's home, which the image does
// not persist (#18960).
const bakedWorkingDir =
  buildStage?.match(/^ARG VITE_WORKING_DIR="?([^"\n]+)"?$/m)?.[1] ?? "";

describe("Docker workspace persistence", () => {
  it("bakes an absolute working dir inside a persisted volume", () => {
    const volumes = JSON.parse(
      dockerfile.match(/^VOLUME (\[.+\])$/m)?.[1] ?? "[]",
    ) as string[];

    expect(path.posix.isAbsolute(bakedWorkingDir)).toBe(true);
    expect(
      volumes.some((volume) => bakedWorkingDir.startsWith(`${volume}/`)),
    ).toBe(true);
  });

  it("pre-creates the baked working dir in the final image", () => {
    const created = stages
      .at(-1)
      ?.match(/^RUN mkdir -p ([^&]+)&&/m)?.[1]
      .trim()
      .split(/[\s\\]+/);

    expect(created).toContain(bakedWorkingDir);
  });
});
