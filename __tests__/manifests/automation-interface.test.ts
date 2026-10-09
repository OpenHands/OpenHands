import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createInterfaceManifest,
  createInterfaceManifestWithSubPages,
} from "./manifest-test-data";

/**
 * The seam resolves its source once at module load, so each case installs its
 * candidate with a module mock and imports the seam fresh.
 */
async function loadSeam(candidate: unknown) {
  vi.resetModules();
  vi.doMock("#/manifests/manifest-sources", () => ({
    AUTOMATION_INTERFACE_CANDIDATE: candidate,
  }));
  return import("#/manifests/automation-interface");
}

afterEach(() => {
  vi.doUnmock("#/manifests/manifest-sources");
  vi.restoreAllMocks();
});

describe("the automation interface seam", () => {
  it("has no interface at all when the package publishes no manifest", async () => {
    // Arrange — the pinned package predates the interface export.
    const seam = await loadSeam(undefined);

    // Act & Assert — the gate is closed, and the accessors behind it refuse
    // rather than invent copy the host does not own. Routes are the host's
    // own registrations, so they answer either way.
    expect(seam.hasAutomationInterface()).toBe(false);
    expect(() => seam.getInterfaceCopy()).toThrow(
      /No automation interface manifest is admitted/,
    );
    expect(() => seam.getAutomationEndpoint("createPrompt")).toThrow();
    expect(() => seam.getAttributeSpec("timeout")).toThrow();
    expect(() => seam.getImportExportSpec()).toThrow();
    expect(() => seam.getFeaturedAutomationIds()).toThrow();
    expect({
      listPath: seam.automationListPath(),
      setupPath: seam.automationSetupPath("github-pr-reviewer"),
      detailPath: seam.automationDetailPath("a/b"),
    }).toEqual({
      listPath: "/automations",
      setupPath: "/automations/new/github-pr-reviewer",
      detailPath: "/automations/a%2Fb",
    });
  });

  it("classifies pathnames inside and outside the automation surface", async () => {
    // Arrange — routes are the host's own registrations, so the matcher
    // answers with or without an admitted manifest.
    const seam = await loadSeam(undefined);

    // Act & Assert
    expect({
      list: seam.isAutomationsRoute("/automations"),
      detail: seam.isAutomationsRoute("/automations/abc-123"),
      templates: seam.isAutomationsRoute("/automations/templates"),
      home: seam.isAutomationsRoute("/"),
      conversation: seam.isAutomationsRoute("/conversations/abc-123"),
      lookalike: seam.isAutomationsRoute("/automations-foo"),
    }).toEqual({
      list: true,
      detail: true,
      templates: true,
      home: false,
      conversation: false,
      lookalike: false,
    });
  });

  it("serves an admitted manifest's data", async () => {
    // Arrange
    const seam = await loadSeam(createInterfaceManifest());

    // Act & Assert — copy, attribute specs, the import/export envelope, and
    // the id lists all come from the manifest.
    expect(seam.hasAutomationInterface()).toBe(true);
    expect({
      listTitle: seam.getInterfaceCopy().listTitle,
      sidebarLabel: seam.getInterfaceCopy().sidebarLabel,
      editTitle: seam.getInterfaceCopy().editTitle,
      nameLabel: seam.getAttributeSpec("name").label,
      timeoutMax: seam.getAttributeSpec("timeout").max,
      importExport: seam.getImportExportSpec(),
      docsUrl: seam.getAutomationsDocsUrl(),
      featured: [...seam.getFeaturedAutomationIds()],
      responders: [...seam.getResponderIntegrationIds()],
    }).toEqual({
      listTitle: "Widget automations",
      sidebarLabel: "Automate everything",
      editTitle: "Edit the widget automation",
      nameLabel: "Widget name",
      timeoutMax: 900,
      importExport: {
        fileKind: "widget",
        fileVersion: 1,
        filenameSuffix: ".widget.json",
        importDefaults: {
          repoProvider: "gitlab",
          placeholderEventSource: "widget-import",
        },
      },
      docsUrl: "https://docs.openhands.dev/widgets",
      featured: ["github-pr-reviewer"],
      responders: ["github"],
    });
  });

  it("hides an attribute the admitted manifest does not declare", async () => {
    // Arrange — the factory manifest declares only `name` and `timeout`.
    const seam = await loadSeam(createInterfaceManifest());

    // Act
    const spec = seam.getAttributeSpec("prompt");

    // Assert
    expect(spec.present).toBe(false);
  });

  it("leaves the sub-page surface absent when the manifest declares none", async () => {
    // Arrange — an admitted manifest without the sub-page group. The host has
    // no defaults for this surface, so nothing stands in.
    const seam = await loadSeam(createInterfaceManifest());

    // Act & Assert
    expect({
      subPages: seam.getSubPagesSpec(),
      dashboard: seam.getDashboardSpec(),
      templates: seam.getTemplatesPageSpec(),
    }).toEqual({ subPages: null, dashboard: null, templates: null });
  });

  it("serves the declared sub-page surface with routes resolved", async () => {
    // Arrange
    const seam = await loadSeam(createInterfaceManifestWithSubPages());

    // Act
    const subPages = seam.getSubPagesSpec();
    const dashboard = seam.getDashboardSpec();
    const templates = seam.getTemplatesPageSpec();

    // Assert — navigation items carry the routes the manifest owns, and the
    // dashboard spec returns the four sections whole.
    expect({
      subPages,
      templatesPath: seam.automationTemplatesPath(),
      sortDefault: dashboard?.sort.default,
      tileMetrics: dashboard?.overview.tiles.map((tile) => tile.metric),
      healthyLabel: dashboard?.insights.health.healthy,
      templates,
    }).toEqual({
      subPages: [
        {
          page: "list",
          to: "/automations",
          label: "Widget dashboard",
          icon: "layout-dashboard",
        },
        {
          page: "templates",
          to: "/automations/templates",
          label: "Widget templates",
          icon: "sparkles",
        },
      ],
      templatesPath: "/automations/templates",
      sortDefault: "name",
      tileMetrics: [
        "automations",
        "needs-attention",
        "total-runs",
        "average-duration",
      ],
      healthyLabel: "Fine",
      templates: {
        title: "Widget templates",
        description: "Pick a proven widget to start from.",
      },
    });
  });

  it("stays unavailable, loudly, when a manifest fails admission", async () => {
    // Arrange
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const seam = await loadSeam(
      createInterfaceManifest({ version: "2.0" as "1.0" }),
    );

    // Assert — a rejected manifest leaves the surface closed, exactly as an
    // absent one does, and the rejection is reported rather than silent.
    expect({
      available: seam.hasAutomationInterface(),
      warned: warn.mock.calls.length,
    }).toEqual({ available: false, warned: 1 });
    expect(() => seam.getInterfaceCopy()).toThrow();
  });
});
