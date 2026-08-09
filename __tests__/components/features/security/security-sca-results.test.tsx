import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SecurityScaResults } from "#/components/features/security/security-sca-results";
import type { ScaScanResult } from "#/types/security-scan";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "en" },
    t: (key: string, options?: { count?: number }) => {
      if (key.includes("SCA_FINDINGS_COUNT") && options?.count != null) {
        return `${options.count} vulnerable package(s)`;
      }
      const labels: Record<string, string> = {
        "SECURITY$SCA_EMPTY_STATE": "Run an SCA scan to analyze dependencies.",
        "SECURITY$SCA_NO_FINDINGS": "No vulnerable dependencies found",
        "SECURITY$SEVERITY_HIGH": "High",
        "SECURITY$SEVERITY_MEDIUM": "Medium",
        "SECURITY$SEVERITY_LOW": "Low",
        "SECURITY$SEVERITY_INFO": "Info",
      };
      return labels[key] ?? key;
    },
  }),
}));

const sampleResult: ScaScanResult = {
  tool: "dependency-track",
  scannedAt: "2026-08-09T00:00:00.000Z",
  findings: [
    {
      id: "finding-1",
      packageName: "lodash",
      packageVersion: "4.17.20",
      purl: "pkg:npm/lodash@4.17.20",
      cveId: "CVE-2021-23337",
      severity: "HIGH",
      description: "Prototype pollution",
    },
    {
      id: "finding-2",
      packageName: "axios",
      packageVersion: "0.21.0",
      purl: "pkg:npm/axios@0.21.0",
      cveId: "CVE-2021-3749",
      severity: "MEDIUM",
      description: "SSRF risk",
    },
  ],
};

describe("SecurityScaResults", () => {
  it("shows the empty state before any scan runs", () => {
    render(<SecurityScaResults result={null} />);

    expect(screen.getByTestId("sca-empty-state")).toBeInTheDocument();
    expect(
      screen.getByText("Run an SCA scan to analyze dependencies."),
    ).toBeInTheDocument();
  });

  it("shows a no-findings message when the scan returns zero issues", () => {
    render(
      <SecurityScaResults result={{ ...sampleResult, findings: [] }} />,
    );

    expect(screen.getByTestId("sca-no-findings")).toBeInTheDocument();
    expect(
      screen.getByText("No vulnerable dependencies found"),
    ).toBeInTheDocument();
  });

  it("renders findings sorted by severity", () => {
    render(<SecurityScaResults result={sampleResult} />);

    expect(screen.getByTestId("sca-results")).toBeInTheDocument();
    expect(screen.getByText("2 vulnerable package(s)")).toBeInTheDocument();
    expect(screen.getByText("Prototype pollution")).toBeInTheDocument();
    expect(screen.getByText("lodash@4.17.20")).toBeInTheDocument();
  });
});
