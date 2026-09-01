import { describe, expect, it } from "vitest";
import type { CanvasExtensionManifest } from "#/types/canvas-extension";
import {
  getCanvasExtensionIconPath,
  isValidCanvasExtensionIconPath,
} from "./canvas-extension-icon";

describe("isValidCanvasExtensionIconPath", () => {
  it("accepts a relative SVG path with subdirectories", () => {
    expect(isValidCanvasExtensionIconPath("assets/pulse.svg")).toBe(true);
  });

  it("accepts a relative path with a leading ./ segment", () => {
    expect(isValidCanvasExtensionIconPath("./icon.svg")).toBe(true);
  });

  it("accepts an SVG extension case-insensitively", () => {
    expect(isValidCanvasExtensionIconPath("assets/ICON.SVG")).toBe(true);
  });

  it("rejects an empty or whitespace-only path", () => {
    expect(isValidCanvasExtensionIconPath("")).toBe(false);
    expect(isValidCanvasExtensionIconPath("   ")).toBe(false);
  });

  it("rejects a non-SVG path", () => {
    expect(isValidCanvasExtensionIconPath("assets/pulse.png")).toBe(false);
    expect(isValidCanvasExtensionIconPath("assets/pulse")).toBe(false);
  });

  it("rejects absolute and URL-style paths", () => {
    expect(isValidCanvasExtensionIconPath("/assets/pulse.svg")).toBe(false);
    expect(isValidCanvasExtensionIconPath("C:/assets/pulse.svg")).toBe(false);
    expect(
      isValidCanvasExtensionIconPath("https://example.com/pulse.svg"),
    ).toBe(false);
    expect(isValidCanvasExtensionIconPath("data:image/svg+xml,<svg/>")).toBe(
      false,
    );
  });

  it("rejects parent-directory traversal in any form", () => {
    expect(isValidCanvasExtensionIconPath("../pulse.svg")).toBe(false);
    expect(isValidCanvasExtensionIconPath("assets/../../pulse.svg")).toBe(
      false,
    );
  });

  it("rejects backslashes and percent-encoding", () => {
    expect(isValidCanvasExtensionIconPath("assets\\pulse.svg")).toBe(false);
    expect(isValidCanvasExtensionIconPath("assets/..%2Fpulse.svg")).toBe(false);
  });
});

describe("getCanvasExtensionIconPath", () => {
  it("returns the trimmed declared icon when valid", () => {
    const manifest: CanvasExtensionManifest = {
      schema_version: 1,
      name: "demo",
      version: "0.1.0",
      icon: "  assets/pulse.svg  ",
      entrypoint: "extension.js",
    };
    expect(getCanvasExtensionIconPath(manifest)).toBe("assets/pulse.svg");
  });

  it("returns null when the manifest has no icon", () => {
    const manifest: CanvasExtensionManifest = {
      schema_version: 1,
      name: "demo",
      version: "0.1.0",
      entrypoint: "extension.js",
    };
    expect(getCanvasExtensionIconPath(manifest)).toBeNull();
  });

  it("returns null for a null or undefined manifest", () => {
    expect(getCanvasExtensionIconPath(null)).toBeNull();
    expect(getCanvasExtensionIconPath(undefined)).toBeNull();
  });

  it("returns null when the declared icon is invalid", () => {
    const manifest: CanvasExtensionManifest = {
      schema_version: 1,
      name: "demo",
      version: "0.1.0",
      icon: "/etc/passwd",
      entrypoint: "extension.js",
    };
    expect(getCanvasExtensionIconPath(manifest)).toBeNull();
  });
});
