import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AddPluginModal } from "#/components/features/plugins/add-plugin-modal";
import PluginsManagementService from "#/api/plugins-management-service";
import { type InstalledPluginInfo } from "#/api/plugins-management-service";
import {
  buildPluginsViewModel,
  matchesPluginStatus,
} from "#/components/features/plugins/build-plugins-view-model";
import PluginsService from "#/api/plugins-service";
import {
  type MarketplacePlugin,
  type LocalPlugin,
} from "#/api/plugins-service";
import { PluginFilesSection } from "#/components/features/plugins/plugin-files-section";
import {
  isPluginSelected,
  marketplacePluginToSpec,
  matchesPluginPickerSearch,
  pluginSpecKey,
  togglePluginSelection,
} from "#/components/features/plugins/plugin-spec-identity";
import { PluginsToolbar } from "#/components/features/plugins/plugins-toolbar";
import { renderWithProviders } from "test-utils";

describe("add-plugin-modal", () => {
  function renderAddPluginModal(onClose = vi.fn()) {
    render(<AddPluginModal onClose={onClose} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          {children}
        </QueryClientProvider>
      ),
    });
    return { onClose };
  }

  describe("AddPluginModal", () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it("installs the plugin with the entered source on submit", async () => {
      const user = userEvent.setup();
      const installSpy = vi
        .spyOn(PluginsManagementService, "installPlugin")
        .mockResolvedValue({
          name: "weather",
          version: "1.0.0",
          description: "",
          enabled: true,
          source: "github:acme/weather",
          resolved_ref: null,
          repo_path: null,
          installed_at: "2026-06-01T00:00:00Z",
          install_path: "/home/.openhands/plugins/installed/weather",
        });

      renderAddPluginModal();
      await user.type(
        screen.getByTestId("add-plugin-source-input"),
        "github:acme/weather",
      );
      await user.click(screen.getByTestId("add-plugin-submit"));

      await waitFor(() =>
        expect(installSpy).toHaveBeenCalledWith({
          source: "github:acme/weather",
          ref: null,
          repo_path: null,
        }),
      );
    });

    it("disables the submit button while the source field is empty", () => {
      renderAddPluginModal();

      expect(screen.getByTestId("add-plugin-submit")).toBeDisabled();
    });
  });
});

describe("build-plugins-view-model", () => {
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
});

describe("plugin-files-section", () => {
  function renderFilesSection(files: string[] = ["README.md"]) {
    return render(
      <PluginFilesSection basePath="/plugins/demo" files={files} />,
      {
        wrapper: ({ children }) => (
          <QueryClientProvider
            client={
              new QueryClient({ defaultOptions: { queries: { retry: false } } })
            }
          >
            {children}
          </QueryClientProvider>
        ),
      },
    );
  }

  describe("PluginFilesSection", () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it("fetches and shows a file's content when the file is clicked", async () => {
      const user = userEvent.setup();
      const contentSpy = vi
        .spyOn(PluginsService, "getPluginFileContent")
        .mockResolvedValue({ kind: "text", text: "# Demo readme" });

      renderFilesSection();
      await user.click(screen.getByTestId("file-tree-file-README.md"));

      await waitFor(() =>
        expect(screen.getByTestId("plugin-file-content")).toHaveTextContent(
          "Demo readme",
        ),
      );
      expect(contentSpy).toHaveBeenCalledWith("/plugins/demo", "README.md");
    });

    it("closes the content viewer when the selected file is clicked again", async () => {
      const user = userEvent.setup();
      vi.spyOn(PluginsService, "getPluginFileContent").mockResolvedValue({
        kind: "text",
        text: "# Demo readme",
      });

      renderFilesSection();
      await user.click(screen.getByTestId("file-tree-file-README.md"));
      await screen.findByTestId("plugin-file-content");
      await user.click(screen.getByTestId("file-tree-file-README.md"));

      expect(
        screen.queryByTestId("plugin-file-content"),
      ).not.toBeInTheDocument();
    });

    it("shows a binary notice instead of content for binary files", async () => {
      const user = userEvent.setup();
      vi.spyOn(PluginsService, "getPluginFileContent").mockResolvedValue({
        kind: "binary",
        text: null,
      });

      renderFilesSection(["logo.png"]);
      await user.click(screen.getByTestId("file-tree-file-logo.png"));

      expect(
        await screen.findByText("FILES$BINARY_FALLBACK"),
      ).toBeInTheDocument();
    });

    it("shows a load error when fetching the file fails", async () => {
      const user = userEvent.setup();
      vi.spyOn(PluginsService, "getPluginFileContent").mockRejectedValue(
        new Error("unreachable"),
      );

      renderFilesSection();
      await user.click(screen.getByTestId("file-tree-file-README.md"));

      expect(await screen.findByText("FILES$LOAD_ERROR")).toBeInTheDocument();
    });
  });
});

describe("plugin-spec-identity", () => {
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
      expect(matchesPluginPickerSearch(catalogPlugin, "nonexistent")).toBe(
        false,
      );
    });
  });
});

describe("plugins-toolbar", () => {
  function renderToolbar() {
    const onSearchChange = vi.fn();
    const onStatusFilterChange = vi.fn();
    renderWithProviders(
      <PluginsToolbar
        search=""
        onSearchChange={onSearchChange}
        statusFilter="all"
        onStatusFilterChange={onStatusFilterChange}
      />,
    );
    return { onSearchChange, onStatusFilterChange };
  }

  describe("PluginsToolbar", () => {
    it("stacks search above status filters below the md breakpoint", () => {
      renderToolbar();

      const toolbar = screen.getByTestId("plugins-toolbar");
      expect(toolbar.className).toContain("flex-col");
      expect(toolbar.className).toContain("md:flex-row");

      const search = screen.getByTestId("plugins-search-input");
      expect(search.parentElement?.className).toContain("w-full");

      const filters = screen.getByTestId("plugins-status-filter");
      const filterClasses = filters.className.split(/\s+/);
      expect(filterClasses).toContain("flex-wrap");
      expect(filterClasses).toContain("md:flex-nowrap");
      expect(filterClasses).not.toContain("shrink-0");
      expect(filterClasses).toContain("md:shrink-0");
    });

    it("keeps every status filter reachable and still filters with search", async () => {
      const user = userEvent.setup();
      const { onSearchChange, onStatusFilterChange } = renderToolbar();

      for (const value of ["all", "installed", "available", "local"]) {
        expect(
          screen.getByTestId(`plugins-filter-${value}`),
        ).toBeInTheDocument();
      }

      await user.click(screen.getByTestId("plugins-filter-installed"));
      expect(onStatusFilterChange).toHaveBeenCalledWith("installed");

      await user.type(screen.getByTestId("plugins-search-input"), "git");
      expect(onSearchChange).toHaveBeenCalled();
    });
  });
});
