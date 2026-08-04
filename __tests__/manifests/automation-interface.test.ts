import { afterEach, describe, expect, it, vi } from "vitest";
import { createInterfaceManifest } from "./manifest-test-data";

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
  it("serves the host defaults when the package publishes no manifest", async () => {
    // Arrange — the pinned package predates the interface export.
    const seam = await loadSeam(undefined);

    // Act & Assert — copy defers to host i18n; every other datum is the
    // host's own literal, byte-identical to pre-seam behavior.
    expect({
      listTitle: seam.getInterfaceCopy().listTitle,
      sidebarLabel: seam.getInterfaceCopy().sidebarLabel,
      listPath: seam.automationListPath(),
      setupPath: seam.automationSetupPath("github-pr-reviewer"),
      detailPath: seam.automationDetailPath("a/b"),
      createEndpoint: seam.getAutomationEndpoint("createPrompt"),
      dispatchEndpoint: seam.getAutomationIdEndpoint("dispatch", "id-1"),
      timeoutMax: seam.getAttributeSpec("timeout").max,
      filenameSuffix: seam.getImportExportSpec().filenameSuffix,
      featured: [...seam.getFeaturedAutomationIds()],
      responders: [...seam.getResponderIntegrationIds()],
    }).toEqual({
      listTitle: null,
      sidebarLabel: null,
      listPath: "/automations",
      setupPath: "/automations/new/github-pr-reviewer",
      detailPath: "/automations/a%2Fb",
      createEndpoint: "/v1/preset/prompt",
      dispatchEndpoint: "/v1/id-1/dispatch",
      timeoutMax: null,
      filenameSuffix: ".automation.json",
      featured: [
        "github-pr-reviewer",
        "github-repo-monitor",
        "slack-channel-monitor",
      ],
      responders: ["github", "slack"],
    });
  });

  it("serves an admitted manifest's data instead of the defaults", async () => {
    // Arrange
    const seam = await loadSeam(createInterfaceManifest());

    // Act & Assert — copy, attribute specs, the import/export envelope, and
    // the id lists all come from the manifest.
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

  it("falls back to the defaults, loudly, when a manifest fails admission", async () => {
    // Arrange
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const seam = await loadSeam(
      createInterfaceManifest({ version: "2.0" as "1.0" }),
    );

    // Act
    const copy = seam.getInterfaceCopy();

    // Assert — nothing from the rejected manifest leaks through, and the
    // rejection is reported rather than silent.
    expect({
      listTitle: copy.listTitle,
      warned: warn.mock.calls.length,
    }).toEqual({ listTitle: null, warned: 1 });
  });
});
