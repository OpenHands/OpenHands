import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createManifestRegistry } from "#/manifests/manifest-registry";
import { createManifest, createManifestWith } from "./manifest-test-data";

describe("createManifestRegistry", () => {
  beforeEach(() => {
    // Rejections are reported for the manifest author, not the user.
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("resolves a manifest from the route it declares", () => {
    // Arrange
    const manifest = createManifest({
      routes: [{ path: "/ext/widget/new", page: "setup" }],
    });
    const registry = createManifestRegistry([manifest]);

    // Act
    const resolved = registry.findByRoutePath("/ext/widget/new/");

    // Assert
    expect(resolved).toBe(manifest);
  });

  it("claims no route the manifests did not declare", () => {
    // Arrange
    const registry = createManifestRegistry([createManifest()]);

    // Act
    const resolved = registry.findByRoutePath("/somewhere-else");

    // Assert
    expect(resolved).toBeNull();
  });

  it("drops a manifest that fails admission instead of rendering part of it", () => {
    // Arrange
    const rejected = createManifestWith({ manifestVersion: "2.0" });

    // Act
    const registry = createManifestRegistry([rejected, createManifest()]);

    // Assert
    expect(registry.manifests).toHaveLength(1);
  });

  it("keeps the first claim on a route so a later manifest cannot hijack it", () => {
    // Arrange
    const first = createManifest({ id: "first" });
    const second = createManifest({ id: "second" });

    // Act
    const registry = createManifestRegistry([first, second]);

    // Assert
    expect(registry.findByRoutePath("/ext/widget")).toBe(first);
  });
});
