import { describe, expect, it } from "vitest";
import type { MarketplacePlugin } from "#/api/plugins-service";
import {
  isPluginSelected,
  marketplacePluginToSpec,
  matchesPluginPickerSearch,
  pluginSpecKey,
  togglePluginSelection,
} from "#/components/features/plugins/plugin-spec-identity";

const catalogPlugin: MarketplacePlugin = {
  name: "city-weather",
  description: "Weather plugin",
  source: "github:OpenHands/extensions",
  ref: null,
  repo_path: "plugins/city-weather",
  installed: false,
};

describe("plugin-spec-identity", () => {
  it("maps a catalog entry to attachable coordinates and drops parameters", () => {
    // Arrange: a catalog entry that also carries non-coordinate fields.
    // Act
    const spec = marketplacePluginToSpec(catalogPlugin);

    // Assert: only source/ref/repo_path survive — no name/description/parameters.
    expect(spec).toEqual({
      source: "github:OpenHands/extensions",
      ref: null,
      repo_path: "plugins/city-weather",
    });
  });

  it("treats missing and explicit-null coordinates as the same identity", () => {
    // Arrange / Act / Assert: an undefined ref and a null ref hash equally,
    // so a catalog entry matches a stored spec regardless of which it used.
    expect(pluginSpecKey({ source: "github:o/a" })).toBe(
      pluginSpecKey({ source: "github:o/a", ref: null, repo_path: null }),
    );
  });

  it("adds an unselected plugin to the selection", () => {
    // Arrange
    const selected = togglePluginSelection([], catalogPlugin);

    // Assert
    expect(selected).toEqual([
      {
        source: "github:OpenHands/extensions",
        ref: null,
        repo_path: "plugins/city-weather",
      },
    ]);
  });

  it("removes a plugin that is already selected", () => {
    // Arrange: the selection already contains the plugin's coordinates.
    const existing = marketplacePluginToSpec(catalogPlugin);

    // Act: toggling the same plugin again removes it.
    const selected = togglePluginSelection([existing], catalogPlugin);

    // Assert
    expect(selected).toEqual([]);
  });

  it("reports whether a plugin's coordinates are in the selection", () => {
    // Arrange
    const selection = [marketplacePluginToSpec(catalogPlugin)];

    // Assert: present coordinates match; a different repo_path does not.
    expect(isPluginSelected(selection, catalogPlugin)).toBe(true);
    expect(
      isPluginSelected(selection, {
        ...catalogPlugin,
        repo_path: "plugins/other",
      }),
    ).toBe(false);
  });

  it("filters case-insensitively by visible fields and matches everything when empty", () => {
    // Assert: empty query keeps all; a query matches across name/description/source.
    expect(matchesPluginPickerSearch(catalogPlugin, "")).toBe(true);
    expect(matchesPluginPickerSearch(catalogPlugin, "WEATHER")).toBe(true);
    expect(matchesPluginPickerSearch(catalogPlugin, "nonexistent")).toBe(false);
  });
});
