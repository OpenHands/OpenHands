// @vitest-environment node
// Importing the module must not trigger a download: main() only runs when
// the script is executed directly (`node scripts/download-uv.mjs`). If that
// main-guard regressed, this import would hit the GitHub API (and its
// failure handler calls process.exit(1)), killing the worker loudly.
import { describe, expect, it } from "vitest";
import { getPlatformSpec } from "../../scripts/download-uv.mjs";

describe("download-uv module import", () => {
  it("imports as a module without starting a download", () => {
    expect(typeof getPlatformSpec).toBe("function");
  });
});

describe("download-uv getPlatformSpec", () => {
  it("maps darwin arches to uv's Rust-style triplets", () => {
    expect(getPlatformSpec("arm64", "darwin")).toEqual({
      target: "uv-aarch64-apple-darwin",
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    });
    expect(getPlatformSpec("x64", "darwin")).toEqual({
      target: "uv-x86_64-apple-darwin",
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    });
  });

  it("maps linux to the x86_64 gnu target", () => {
    expect(getPlatformSpec("x64", "linux")).toEqual({
      target: "uv-x86_64-unknown-linux-gnu",
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    });
  });

  it("maps win32 to the msvc zip with .exe binaries", () => {
    expect(getPlatformSpec("x64", "win32")).toEqual({
      target: "uv-x86_64-pc-windows-msvc",
      ext: "zip",
      binaries: ["uv.exe", "uvx.exe"],
    });
  });

  it("throws for an unsupported platform", () => {
    expect(() => getPlatformSpec("arm64", "sunos")).toThrow(
      /Unsupported platform for uv download: sunos/,
    );
  });
});
