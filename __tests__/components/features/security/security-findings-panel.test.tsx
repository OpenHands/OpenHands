import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SecurityFindingsPanel } from "#/components/features/security/security-findings-panel";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";

const exportMock = vi.fn();

vi.mock("#/utils/security-findings-export", async () => {
  const actual = await vi.importActual<
    typeof import("#/utils/security-findings-export")
  >("#/utils/security-findings-export");
  return {
    ...actual,
    exportSecurityFindings: (...args: unknown[]) => exportMock(...args),
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "en" },
    t: (key: string, options?: { shown?: number; total?: number }) => {
      if (key.includes("FILTERED_COUNT")) {
        return `Showing ${options?.shown} of ${options?.total} finding(s)`;
      }
      const labels: Record<string, string> = {
        "COMMON$SECURITY": "Security",
        "SECURITY$EMPTY_STATE": "Run a scan to find security issues.",
        "SECURITY$NO_FINDINGS": "No security issues found",
        "SECURITY$NO_FILTERED_FINDINGS": "No findings match the current filters",
        "SECURITY$FILTER_TOOL": "Tool",
        "SECURITY$FILTER_TOOL_ALL": "All tools",
        "SECURITY$FILTER_TOOL_SAST": "SAST (OpenGrep)",
        "SECURITY$FILTER_TOOL_SCA": "SCA (Dependency-Track)",
        "SECURITY$FILTER_SEVERITY": "Severity",
        "SECURITY$FILTER_SEVERITY_ALL": "All severities",
        "SECURITY$FILTER_SEARCH": "Search findings",
        "SECURITY$FILTER_SEARCH_PLACEHOLDER": "Search…",
        "SECURITY$EXPORT": "Export",
        "SECURITY$EXPORT_CSV": "CSV",
        "SECURITY$EXPORT_EXCEL": "Excel",
        "SECURITY$EXPORT_PDF": "PDF",
        "SECURITY$EXPORT_COLUMN_REFERENCE": "Reference",
        "SECURITY$EXPORT_COLUMN_DESCRIPTION": "Description",
        "SECURITY$EXPORT_COLUMN_LOCATION": "Location",
        "SECURITY$SEVERITY_HIGH": "High",
        "SECURITY$SEVERITY_MEDIUM": "Medium",
        "SECURITY$SEVERITY_LOW": "Low",
        "SECURITY$SEVERITY_INFO": "Info",
      };
      return labels[key] ?? key;
    },
  }),
}));

const sast: SecurityScanResult = {
  tool: "opengrep",
  scannedAt: "2026-08-09T00:00:00.000Z",
  findings: [
    {
      id: "1",
      ruleId: "js.eval",
      message: "Detected use of eval",
      severity: "ERROR",
      filePath: "src/app.js",
      startLine: 10,
      startCol: 1,
      endLine: 10,
      endCol: 5,
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
      severity: "MEDIUM",
      description: "Prototype pollution",
    },
  ],
};

describe("SecurityFindingsPanel", () => {
  it("shows the empty state before any scan", () => {
    render(<SecurityFindingsPanel sastResult={null} scaResult={null} />);
    expect(screen.getByTestId("security-empty-state")).toBeInTheDocument();
  });

  it("filters by tool and exports the visible rows", async () => {
    const user = userEvent.setup();
    render(<SecurityFindingsPanel sastResult={sast} scaResult={sca} />);

    expect(screen.getByTestId("security-filtered-count")).toHaveTextContent(
      "Showing 2 of 2 finding(s)",
    );

    await user.selectOptions(screen.getByTestId("security-filter-tool"), "sca");
    expect(screen.getByText("Prototype pollution")).toBeInTheDocument();
    expect(screen.queryByText("Detected use of eval")).not.toBeInTheDocument();
    expect(screen.getByTestId("security-filtered-count")).toHaveTextContent(
      "Showing 1 of 2 finding(s)",
    );

    await user.click(screen.getByTestId("security-export-csv"));
    expect(exportMock).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          tool: "sca",
          reference: "CVE-2021-23337",
        }),
      ]),
      "csv",
      expect.any(Object),
    );
  });
});
