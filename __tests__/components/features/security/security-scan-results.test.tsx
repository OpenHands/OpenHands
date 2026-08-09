import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SecurityScanResults } from "#/components/features/security/security-scan-results";
import type { SecurityScanResult } from "#/types/security-scan";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "en" },
    t: (key: string, options?: { count?: number }) => {
      if (key.includes("FINDINGS_COUNT") && options?.count != null) {
        return `${options.count} issue(s) found`;
      }
      const labels: Record<string, string> = {
        "SECURITY$EMPTY_STATE": "Run a scan to find security issues.",
        "SECURITY$NO_FINDINGS": "No security issues found",
        "SECURITY$SEVERITY_HIGH": "High",
        "SECURITY$SEVERITY_MEDIUM": "Medium",
        "SECURITY$SEVERITY_LOW": "Low",
        "SECURITY$SEVERITY_INFO": "Info",
      };
      return labels[key] ?? key;
    },
  }),
}));

const sampleResult: SecurityScanResult = {
  tool: "opengrep",
  scannedAt: "2026-08-09T00:00:00.000Z",
  findings: [
    {
      id: "rule:src/app.py:10:5:0",
      ruleId: "python.lang.security.audit.eval-detected",
      message: "Detected use of eval",
      severity: "ERROR",
      filePath: "src/app.py",
      startLine: 10,
      startCol: 5,
      endLine: 10,
      endCol: 20,
    },
    {
      id: "rule:src/utils.ts:3:1:1",
      ruleId: "typescript.lang.security.audit",
      message: "Insecure pattern",
      severity: "WARNING",
      filePath: "src/utils.ts",
      startLine: 3,
      startCol: 1,
      endLine: 3,
      endCol: 12,
    },
  ],
};

describe("SecurityScanResults", () => {
  it("shows the empty state before any scan runs", () => {
    render(<SecurityScanResults result={null} />);

    expect(screen.getByTestId("security-empty-state")).toBeInTheDocument();
    expect(
      screen.getByText("Run a scan to find security issues."),
    ).toBeInTheDocument();
  });

  it("shows a no-findings message when the scan returns zero issues", () => {
    render(
      <SecurityScanResults
        result={{ ...sampleResult, findings: [] }}
      />,
    );

    expect(screen.getByTestId("security-no-findings")).toBeInTheDocument();
    expect(screen.getByText("No security issues found")).toBeInTheDocument();
  });

  it("renders findings sorted by severity", () => {
    render(<SecurityScanResults result={sampleResult} />);

    expect(screen.getByTestId("security-results")).toBeInTheDocument();
    expect(screen.getByText("2 issue(s) found")).toBeInTheDocument();
    expect(screen.getByText("Detected use of eval")).toBeInTheDocument();
    expect(screen.getByText("Insecure pattern")).toBeInTheDocument();
    expect(screen.getByText("src/app.py:10:5")).toBeInTheDocument();
  });
});
