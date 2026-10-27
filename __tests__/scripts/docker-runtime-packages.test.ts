// @vitest-environment node
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const dockerfilePath = path.join(repoRoot, "docker/Dockerfile");
const temporaryDirectories: string[] = [];

function readDockerfile(): string {
  return readFileSync(dockerfilePath, "utf-8");
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("Docker runtime package metadata", () => {
  it("keeps the Canvas snapshot at least as new as its agent-server base", () => {
    const match = readDockerfile().match(
      /^ARG DEBIAN_SNAPSHOT=(\d{8}T\d{6}Z)$/m,
    );

    expect(match?.[1]).toBeTruthy();
    expect(match?.[1].localeCompare("20260920T000000Z")).toBeGreaterThanOrEqual(
      0,
    );
  });

  it("selects a snapshot after the seven-day observation period", () => {
    const directory = mkdtempSync(path.join(tmpdir(), "canvas-dockerfile-"));
    temporaryDirectories.push(directory);
    const candidate = path.join(directory, "Dockerfile");
    writeFileSync(
      candidate,
      "FROM debian:trixie-slim\nARG DEBIAN_SNAPSHOT=20260913T000000Z\n",
    );

    execFileSync(
      "python3",
      [
        path.join(repoRoot, ".github/scripts/update_debian_snapshot.py"),
        "--dockerfile",
        candidate,
        "--now",
        "2026-10-07T23:03:00Z",
        "--skip-network-check",
      ],
      { stdio: "pipe" },
    );

    expect(readFileSync(candidate, "utf-8")).toContain(
      "ARG DEBIAN_SNAPSHOT=20260930T000000Z",
    );
  });

  it("does not retain PostgreSQL development packages at runtime", () => {
    const dockerfile = readDockerfile();

    expect(dockerfile).not.toMatch(/apt-get install[^;]*libpq(?:-dev|5)/s);
    expect(dockerfile.indexOf("apt-get upgrade")).toBeLessThan(
      dockerfile.indexOf('uv pip install --system "openhands-automation'),
    );
  });
});
