import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import { getIntegrationIds } from "#/utils/automation-catalog";

describe("OpenHands extensions catalogs", () => {
  it("loads recommended automations from @openhands/extensions", () => {
    expect(AUTOMATION_CATALOG.length).toBeGreaterThan(0);

    const knownMcpIds = new Set(INTEGRATION_CATALOG.map((entry) => entry.id));
    for (const automation of AUTOMATION_CATALOG) {
      const integrationIds = getIntegrationIds(automation);
      expect(integrationIds.every((id) => knownMcpIds.has(id))).toBe(true);
    }

    // Declaring none is legitimate — `news-digest` connects to nothing — so the
    // resolution above is only worth asserting while some entry still declares
    // one. Without this the loop above would pass over an empty catalog.
    expect(
      AUTOMATION_CATALOG.some(
        (automation) => getIntegrationIds(automation).length > 0,
      ),
    ).toBe(true);
  });

  it("admits every setup experience the automation catalog ships", () => {
    // Arrange — the pinned package is the whole source of setup manifests, and
    // a shipped one that fails admission is dropped silently.
    const shipped = AUTOMATION_CATALOG.filter(
      (automation) => !!automation.setup,
    );
    expect(shipped.length).toBeGreaterThan(0);

    // Act / Assert
    expect(SETUP_REGISTRY.entries.map((entry) => entry.id)).toEqual(
      shipped.map((automation) => automation.id),
    );
  });
});
