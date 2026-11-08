// @vitest-environment node
// Resolve the bundled runtime dir (uv `bin`, Node `node`) inside a packaged
// Electron app: universal macOS builds ship arch-tagged `<base>-<arch>`
// dirs (picked via process.arch — the executing slice of the universal
// binary), single-arch builds ship the legacy flat `<base>` dir. These
// tests build throwaway <resourcesPath> fixtures to pin the candidate
// order: arch-tagged first, legacy flat second, null when neither exists.
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { resolveBundledRuntimeDir } from "../../../electron/lib/bundled-runtime-dir.mjs";

let resourcesPath: string;

beforeEach(() => {
  resourcesPath = mkdtempSync(join(tmpdir(), "bundled-runtime-dir-"));
});

afterEach(() => {
  rmSync(resourcesPath, { recursive: true, force: true });
});

describe("resolveBundledRuntimeDir", () => {
  it.each(["arm64", "x64"])(
    "returns the arch-tagged dir when only it exists (arch=%s)",
    (arch) => {
      const tagged = join(resourcesPath, `node-${arch}`);
      mkdirSync(tagged);

      expect(resolveBundledRuntimeDir(resourcesPath, "node", arch)).toBe(
        tagged,
      );
    },
  );

  it("prefers the arch-tagged dir when both candidates exist", () => {
    const tagged = join(resourcesPath, "bin-arm64");
    const legacy = join(resourcesPath, "bin");
    mkdirSync(tagged);
    mkdirSync(legacy);

    expect(resolveBundledRuntimeDir(resourcesPath, "bin", "arm64")).toBe(
      tagged,
    );
  });

  it("falls back to the legacy flat dir for single-arch builds", () => {
    const legacy = join(resourcesPath, "bin");
    mkdirSync(legacy);

    expect(resolveBundledRuntimeDir(resourcesPath, "bin", "x64")).toBe(legacy);
  });

  it("returns null when neither candidate exists", () => {
    expect(
      resolveBundledRuntimeDir(resourcesPath, "node", "arm64"),
    ).toBeNull();
  });
});
