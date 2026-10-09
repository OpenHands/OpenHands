import { describe, expect, it, vi } from "vitest";

/**
 * Without an admitted interface manifest there is no automation surface to
 * render — the pages exist only to render what a manifest describes. Each
 * route says so the way the host says it everywhere else: a 404 the layout's
 * error boundary renders.
 */
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  return { ...actual, AUTOMATION_INTERFACE_CANDIDATE: undefined };
});

function statusOfLoader(load: () => unknown): number | null {
  try {
    load();
  } catch (error) {
    return error instanceof Response ? error.status : null;
  }
  return null;
}

describe("the automation routes without an admitted interface manifest", () => {
  it("404s every automation route", async () => {
    // Arrange
    const [list, detail, setup, templates] = await Promise.all([
      import("#/routes/automations-list"),
      import("#/routes/automation-detail"),
      import("#/routes/automation-setup-route"),
      import("#/routes/automation-templates"),
    ]);

    // Act & Assert
    expect({
      list: statusOfLoader(() => list.clientLoader()),
      detail: statusOfLoader(() => detail.clientLoader()),
      setup: statusOfLoader(() =>
        setup.clientLoader({
          params: { automationId: "github-pr-reviewer" },
        } as Parameters<typeof setup.clientLoader>[0]),
      ),
      templates: statusOfLoader(() => templates.clientLoader()),
    }).toEqual({ list: 404, detail: 404, setup: 404, templates: 404 });
  });
});
