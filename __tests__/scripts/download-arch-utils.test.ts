// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  DARWIN_UNIVERSAL_ARCHES,
  UNIVERSAL_ARCH,
  resolveDownloadArches,
  resourceDirName,
} from "../../scripts/download-arch-utils.mjs";

describe("resolveDownloadArches", () => {
  it("returns only the host arch when ELECTRON_ARCH is unset", () => {
    expect(
      resolveDownloadArches({ platform: "darwin", hostArch: "arm64" }),
    ).toEqual(["arm64"]);
    expect(
      resolveDownloadArches({
        platform: "linux",
        hostArch: "x64",
        electronArch: undefined,
      }),
    ).toEqual(["x64"]);
  });

  it("returns both darwin arches for ELECTRON_ARCH=universal on darwin", () => {
    expect(
      resolveDownloadArches({
        platform: "darwin",
        hostArch: "arm64",
        electronArch: "universal",
      }),
    ).toEqual(["arm64", "x64"]);
    // Order is load-bearing: electron-builder.config.mjs maps these onto
    // resources/{node,bin}-<arch> directories.
    expect(DARWIN_UNIVERSAL_ARCHES).toEqual(["arm64", "x64"]);
    expect(UNIVERSAL_ARCH).toBe("universal");
  });

  it("throws for ELECTRON_ARCH=universal on linux", () => {
    expect(() =>
      resolveDownloadArches({
        platform: "linux",
        hostArch: "x64",
        electronArch: "universal",
      }),
    ).toThrow(/universal.*darwin|darwin.*universal/i);
  });

  it("throws for ELECTRON_ARCH=universal on win32", () => {
    expect(() =>
      resolveDownloadArches({
        platform: "win32",
        hostArch: "x64",
        electronArch: "universal",
      }),
    ).toThrow(/universal.*darwin|darwin.*universal/i);
  });

  it("throws for an explicit single arch value like x64 — fail loud instead of bundling a silently-wrong cross-arch runtime", () => {
    expect(() =>
      resolveDownloadArches({
        platform: "darwin",
        hostArch: "arm64",
        electronArch: "x64",
      }),
    ).toThrow(/ELECTRON_ARCH/i);
  });

  it("throws for garbage ELECTRON_ARCH values", () => {
    expect(() =>
      resolveDownloadArches({
        platform: "darwin",
        hostArch: "arm64",
        electronArch: "bogus",
      }),
    ).toThrow(/ELECTRON_ARCH/i);
  });
});

describe("resourceDirName", () => {
  it("suffixed the arch onto the base in multi-arch (universal) mode", () => {
    expect(resourceDirName("node", "arm64", true)).toBe("node-arm64");
    expect(resourceDirName("node", "x64", true)).toBe("node-x64");
    expect(resourceDirName("bin", "arm64", true)).toBe("bin-arm64");
    expect(resourceDirName("bin", "x64", true)).toBe("bin-x64");
  });

  it("keeps the legacy single-arch base directory name", () => {
    expect(resourceDirName("node", "arm64", false)).toBe("node");
    expect(resourceDirName("bin", "x64", false)).toBe("bin");
  });
});
