// @vitest-environment node
// Importing the module must not trigger a download: main() only runs when
// the script is executed directly (`node scripts/download-node.mjs`). If
// that main-guard regressed, this import would kick off a real network
// download (and its failure handler calls process.exit(1)), killing the
// worker loudly.
import { describe, expect, it } from "vitest";
import { getPlatformSpec, resolveVersion } from "../../scripts/download-node.mjs";

describe("download-node module import", () => {
  it("imports as a module without starting a download", () => {
    expect(typeof getPlatformSpec).toBe("function");
    expect(typeof resolveVersion).toBe("function");
  });
});

describe("download-node getPlatformSpec", () => {
  it("maps explicit darwin arches to the matching distribution", () => {
    expect(getPlatformSpec("22.12.0", "darwin", "x64")).toEqual({
      name: "node-v22.12.0-darwin-x64",
      ext: "tar.gz",
    });
    expect(getPlatformSpec("22.12.0", "darwin", "arm64")).toEqual({
      name: "node-v22.12.0-darwin-arm64",
      ext: "tar.gz",
    });
  });

  it("maps linux arches to the matching distribution", () => {
    expect(getPlatformSpec("22.12.0", "linux", "x64")).toEqual({
      name: "node-v22.12.0-linux-x64",
      ext: "tar.gz",
    });
    expect(getPlatformSpec("22.12.0", "linux", "arm64")).toEqual({
      name: "node-v22.12.0-linux-arm64",
      ext: "tar.gz",
    });
  });

  it("maps win32 arches to the zip distribution", () => {
    expect(getPlatformSpec("22.12.0", "win32", "x64")).toEqual({
      name: "node-v22.12.0-win-x64",
      ext: "zip",
    });
    expect(getPlatformSpec("22.12.0", "win32", "arm64")).toEqual({
      name: "node-v22.12.0-win-arm64",
      ext: "zip",
    });
  });

  it("defaults platform/arch to the host values", () => {
    const spec = getPlatformSpec("22.12.0");
    expect(spec.name.startsWith("node-v22.12.0-")).toBe(true);
    expect(["darwin", "linux", "win32"]).toContain(
      spec.name.replace("node-v22.12.0-", "").split("-")[0],
    );
  });

  it("throws for an unsupported platform", () => {
    expect(() => getPlatformSpec("22.12.0", "freebsd", "x64")).toThrow(
      /Unsupported platform for Node download: freebsd\/x64/,
    );
  });
});
