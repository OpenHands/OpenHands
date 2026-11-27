import { describe, expect, it } from "vitest";
import {
  buildPluginsViewModel,
  matchesPluginStatus,
} from "#/components/features/plugins/build-plugins-view-model";
import type { MarketplacePlugin, LocalPlugin } from "#/api/plugins-service";
import type { InstalledPluginInfo } from "#/api/plugins-management-service";

const catalogPlugin: MarketplacePlugin = {
  name: "demo-plugin",
  description: "Catalog description",
  source: "github:OpenHands/extensions",
  ref: null,
  repo_path: "plugins/demo-plugin",
  installed: false,
};

const installedPlugin: InstalledPluginInfo = {
  name: "demo-plugin",
  version: "2.0.0",
  description: "Installed description",
  enabled: false,
  source: "github:OpenHands/extensions",
  resolved_ref: "main",
  repo_path: "plugins/demo-plugin",
  installed_at: "2026-06-01T00:00:00Z",
  install_path: "/home/.openhands/plugins/installed/demo-plugin",
};

const localPlugin: LocalPlugin = {
  name: "ambient-plugin",
  version: "1.0.0",
  description: "Ambient description",
};

describe("buildPluginsViewModel", () => {
  it("merges a plugin present in both the catalog and the installed list into one installed entry", () => {
    const result = buildPluginsViewModel([catalogPlugin], [installedPlugin]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      name: "demo-plugin",
      installed: true,
      inCatalog: true,
      enabled: false,
      version: "2.0.0",
    });
  });

  it("marks a catalog-only plugin as not installed", () => {
    const result = buildPluginsViewModel([catalogPlugin], []);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      name: "demo-plugin",
      installed: false,
      inCatalog: true,
    });
  });

  it("keeps an installed plugin that is absent from the catalog", () => {
    const result = buildPluginsViewModel([], [installedPlugin]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      name: "demo-plugin",
      installed: true,
      inCatalog: false,
      enabled: false,
    });
  });

  it("adds a locally-discovered plugin as a read-only local entry", () => {
    const result = buildPluginsViewModel([], [], [localPlugin]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      name: "ambient-plugin",
      isLocal: true,
      installed: false,
      inCatalog: false,
      version: "1.0.0",
    });
  });

  it("does not add a separate local entry when the plugin is already installed", () => {
    const result = buildPluginsViewModel(
      [],
      [installedPlugin],
      [{ ...localPlugin, name: installedPlugin.name }],
    );

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      name: "demo-plugin",
      installed: true,
      isLocal: false,
    });
  });

  it("classifies a local plugin under the local filter and not the available filter", () => {
    const [local] = buildPluginsViewModel([], [], [localPlugin]);

    expect(matchesPluginStatus(local, "local")).toBe(true);
    expect(matchesPluginStatus(local, "available")).toBe(false);
  });

  it("prefers the installed copy's contents over the catalog's", () => {
    const result = buildPluginsViewModel(
      [
        {
          ...catalogPlugin,
          path: "/cache/plugins/demo-plugin",
          skills: [{ name: "catalog-skill" }],
          files: ["catalog.md"],
        },
      ],
      [
        {
          ...installedPlugin,
          skills: [{ name: "installed-skill" }],
          files: ["installed.md"],
        },
      ],
    );

    expect(result[0]).toMatchObject({
      path: installedPlugin.install_path,
      skills: [{ name: "installed-skill" }],
      files: ["installed.md"],
    });
  });

  it("falls back to the catalog's contents when the installed entry has none", () => {
    const result = buildPluginsViewModel(
      [
        {
          ...catalogPlugin,
          path: "/cache/plugins/demo-plugin",
          skills: [{ name: "catalog-skill" }],
          files: ["catalog.md"],
        },
      ],
      [installedPlugin],
    );

    expect(result[0]).toMatchObject({
      path: "/cache/plugins/demo-plugin",
      skills: [{ name: "catalog-skill" }],
      files: ["catalog.md"],
    });
  });

  it("carries a local plugin's contents into its entry", () => {
    const result = buildPluginsViewModel(
      [],
      [],
      [
        {
          ...localPlugin,
          path: "/home/.agents/plugins/ambient-plugin",
          skills: [{ name: "ambient-skill" }],
          files: ["SKILL.md"],
        },
      ],
    );

    expect(result[0]).toMatchObject({
      path: "/home/.agents/plugins/ambient-plugin",
      skills: [{ name: "ambient-skill" }],
      files: ["SKILL.md"],
    });
  });
});
