import { describe, expect, it, vi } from "vitest";
import {
  getSecurityFindingsExportFilename,
  serializeSecurityFindingsCsv,
  serializeSecurityFindingsExcelXml,
} from "#/utils/security-findings-export";
import type { SecurityFindingViewModel } from "#/utils/security-findings-view";

const labels = {
  title: "Security",
  tool: "Tool",
  severity: "Severity",
  reference: "Reference",
  description: "Description",
  location: "Location",
  toolSast: "SAST (OpenGrep)",
  toolSca: "SCA (Dependency-Track)",
  severityHigh: "High",
  severityMedium: "Medium",
  severityLow: "Low",
  severityInfo: "Info",
};

const rows: SecurityFindingViewModel[] = [
  {
    id: "sast:1",
    tool: "sast",
    severity: "ERROR",
    severityBucket: "high",
    reference: "js.lang.security.eval",
    description: 'Detected use of "eval"',
    location: "src/app.js:10:1",
    filePath: "src/app.js",
  },
];

describe("security findings export", () => {
  it("builds a timestamped filename per format", () => {
    const now = new Date("2026-08-09T12:34:56.000Z");
    expect(getSecurityFindingsExportFilename("csv", now)).toBe(
      "security-findings-2026-08-09-12-34-56.csv",
    );
    expect(getSecurityFindingsExportFilename("excel", now)).toBe(
      "security-findings-2026-08-09-12-34-56.xls",
    );
    expect(getSecurityFindingsExportFilename("pdf", now)).toBe(
      "security-findings-2026-08-09-12-34-56.pdf",
    );
  });

  it("serializes CSV with a UTF-8 BOM and escaped quotes", () => {
    const csv = serializeSecurityFindingsCsv(rows, labels);
    expect(csv.startsWith("\uFEFF")).toBe(true);
    expect(csv).toContain("Tool,Severity,Reference,Description,Location");
    expect(csv).toContain('"Detected use of ""eval"""');
    expect(csv).toContain("SAST (OpenGrep),High,js.lang.security.eval");
  });

  it("serializes SpreadsheetML excel xml", () => {
    const xml = serializeSecurityFindingsExcelXml(rows, labels);
    expect(xml).toContain("Excel.Sheet");
    expect(xml).toContain("Detected use of &quot;eval&quot;");
    expect(xml).toContain("<Worksheet ss:Name=\"Findings\">");
  });
});

describe("exportSecurityFindings pdf path", () => {
  it("opens a print window for PDF export", async () => {
    const print = vi.fn();
    const popup = {
      addEventListener: (event: string, cb: () => void) => {
        if (event === "load") cb();
      },
      focus: vi.fn(),
      print,
    };
    vi.stubGlobal(
      "open",
      vi.fn(() => popup),
    );
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:test"),
      revokeObjectURL: vi.fn(),
    });

    const { exportSecurityFindings } = await import(
      "#/utils/security-findings-export"
    );
    exportSecurityFindings(rows, "pdf", labels);
    expect(print).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
