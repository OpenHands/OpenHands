import { describe, expect, it } from "vitest";
import { mapDependencyTrackFindings } from "#/utils/dependency-track-findings";

describe("mapDependencyTrackFindings", () => {
  it("maps Dependency-Track findings into SCA findings", () => {
    const findings = mapDependencyTrackFindings([
      {
        uuid: "finding-1",
        component: {
          name: "lodash",
          version: "4.17.20",
          purl: "pkg:npm/lodash@4.17.20",
        },
        vulnerability: {
          vulnId: "CVE-2021-23337",
          severity: "HIGH",
          description: "Prototype pollution in lodash",
        },
      },
    ]);

    expect(findings).toHaveLength(1);
    expect(findings[0]).toMatchObject({
      id: "finding-1",
      packageName: "lodash",
      packageVersion: "4.17.20",
      purl: "pkg:npm/lodash@4.17.20",
      cveId: "CVE-2021-23337",
      severity: "HIGH",
      description: "Prototype pollution in lodash",
    });
  });

  it("falls back to INFO severity for unknown values", () => {
    const findings = mapDependencyTrackFindings([
      {
        component: { name: "example" },
        vulnerability: { vulnId: "CVE-0000-0000", severity: "UNKNOWN" },
      },
    ]);

    expect(findings[0]?.severity).toBe("INFO");
  });
});
