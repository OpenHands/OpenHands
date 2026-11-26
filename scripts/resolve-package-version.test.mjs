import { afterEach, describe, expect, it } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { resolvePackageVersion } from "./resolve-package-version.mjs";

let root;
afterEach(() => {
  if (root) rmSync(root, { recursive: true, force: true });
  root = undefined;
});

describe("resolvePackageVersion", () => {
  it("reads the version of the package containing the resolved entry point", () => {
    root = mkdtempSync(join(tmpdir(), "resolved-package-version-"));
    const packageDirectory = join(
      root,
      "node_modules",
      "@openhands",
      "typescript-client",
    );
    const entry = join(packageDirectory, "dist", "index.js");
    mkdirSync(join(packageDirectory, "dist"), { recursive: true });
    writeFileSync(
      join(root, "package.json"),
      JSON.stringify({
        dependencies: { "@openhands/typescript-client": "1.0.0" },
      }),
    );
    writeFileSync(
      join(packageDirectory, "package.json"),
      JSON.stringify({
        name: "@openhands/typescript-client",
        version: "1.64.1",
      }),
    );
    writeFileSync(entry, "");

    expect(resolvePackageVersion("@openhands/typescript-client", entry)).toBe(
      "1.64.1",
    );
  });
});
