import { describe, expect, it } from "vitest";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";
import {
  buildUnifiedSecurityFindings,
  filterSecurityFindings,
} from "#/utils/security-findings-view";

const sast: SecurityScanResult = {
  tool: "opengrep",
  scannedAt: "2026-08-09T00:00:00.000Z",
  findings: [
    {
      id: "1",
      ruleId: "js.lang.security.eval",
      message: "Detected use of eval",
      severity: "ERROR",
      filePath: "src/app.js",
      startLine: 10,
      startCol: 1,
      endLine: 10,
      endCol: 5,
    },
    {
      id: "2",
      ruleId: "js.lang.security.info",
      message: "Informational note",
      severity: "INFO",
      filePath: "src/util.js",
      startLine: 1,
      startCol: 1,
      endLine: 1,
      endCol: 2,
    },
  ],
};

const sca: ScaScanResult = {
  tool: "dependency-track",
  scannedAt: "2026-08-09T00:00:00.000Z",
  findings: [
    {
      id: "a",
      packageName: "lodash",
      packageVersion: "4.17.20",
      purl: "pkg:npm/lodash@4.17.20",
      cveId: "CVE-2021-23337",
      severity: "HIGH",
      description: "Prototype pollution",
    },
  ],
};

describe("buildUnifiedSecurityFindings", () => {
  it("merges SAST and SCA rows sorted by severity", () => {
    const rows = buildUnifiedSecurityFindings(sast, sca);
    expect(rows).toHaveLength(3);
    // HIGH (SCA) ranks above ERROR (SAST); INFO last.
    expect(rows.map((r) => r.tool)).toEqual(["sca", "sast", "sast"]);
    expect(rows[0].severityBucket).toBe("high");
    expect(rows[0].reference).toBe("CVE-2021-23337");
  });
});

describe("filterSecurityFindings", () => {
  const rows = buildUnifiedSecurityFindings(sast, sca);

  it("filters by tool", () => {
    expect(
      filterSecurityFindings(rows, {
        tool: "sca",
        severity: "all",
        query: "",
      }),
    ).toHaveLength(1);
  });

  it("filters by severity bucket", () => {
    expect(
      filterSecurityFindings(rows, {
        tool: "all",
        severity: "info",
        query: "",
      }),
    ).toHaveLength(1);
  });

  it("filters by free-text query across description and reference", () => {
    expect(
      filterSecurityFindings(rows, {
        tool: "all",
        severity: "all",
        query: "CVE-2021",
      }),
    ).toHaveLength(1);
    expect(
      filterSecurityFindings(rows, {
        tool: "all",
        severity: "all",
        query: "eval",
      }),
    ).toHaveLength(1);
  });
});
