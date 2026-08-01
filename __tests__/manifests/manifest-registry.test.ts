import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSetupRegistry } from "#/manifests/manifest-registry";
import {
  createSetup,
  createSetupEntry,
  createSetupEntryWith,
} from "./manifest-test-data";

describe("createSetupRegistry", () => {
  beforeEach(() => {
    // Rejections are reported for the manifest author, not the user.
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("resolves an admitted manifest by its id", () => {
    // Arrange
    const entry = createSetupEntry();
    const registry = createSetupRegistry([entry]);

    // Act
    const resolved = registry.findById("widget-monitor");

    // Assert
    expect(resolved).toBe(entry);
  });

  it("claims no id the catalog did not offer", () => {
    // Arrange
    const registry = createSetupRegistry([createSetupEntry()]);

    // Act
    const resolved = registry.findById("something-else");

    // Assert
    expect(resolved).toBeNull();
  });

  it("passes over a catalog entry that ships no setup experience", () => {
    // Arrange — a card-only entry is the normal case, not a rejection.
    const cardOnly = { id: "card-only", name: "Card only" };

    // Act
    const registry = createSetupRegistry([cardOnly, createSetupEntry()]);

    // Assert
    expect(registry.entries).toHaveLength(1);
  });

  it("drops a manifest that fails admission instead of rendering part of it", () => {
    // Arrange
    const rejected = createSetupEntryWith({
      setup: createSetup({ version: "2.0" as "1.0" }),
    });

    // Act
    const registry = createSetupRegistry([rejected, createSetupEntry()]);

    // Assert
    expect(registry.entries).toHaveLength(1);
  });

  it("keeps the first claim on an id so a later manifest cannot hijack it", () => {
    // Arrange
    const first = createSetupEntry({ name: "First" });
    const second = createSetupEntry({ name: "Second" });

    // Act
    const registry = createSetupRegistry([first, second]);

    // Assert
    expect(registry.findById("widget-monitor")).toBe(first);
  });
});
