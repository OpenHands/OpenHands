import type {
  ScaFinding,
  ScaScanResult,
  SecurityFinding,
  SecurityFindingSeverity,
  SecurityScanResult,
} from "#/types/security-scan";

export type SecurityToolFilter = "all" | "sast" | "sca";

export type SecuritySeverityBucket = "high" | "medium" | "low" | "info";

export type SecuritySeverityFilter = "all" | SecuritySeverityBucket;

export interface SecurityFindingViewModel {
  id: string;
  tool: "sast" | "sca";
  severity: SecurityFindingSeverity;
  severityBucket: SecuritySeverityBucket;
  /** Rule id (SAST) or CVE / vuln id (SCA). */
  reference: string;
  /** Human description (English source; may be translated at render). */
  description: string;
  /** File:line (SAST) or package@version (SCA). */
  location: string;
  filePath?: string;
  packageName?: string;
}

export interface SecurityFindingsFilters {
  tool: SecurityToolFilter;
  severity: SecuritySeverityFilter;
  query: string;
}

export const DEFAULT_SECURITY_FINDINGS_FILTERS: SecurityFindingsFilters = {
  tool: "all",
  severity: "all",
  query: "",
};

export const SECURITY_SEVERITY_ORDER: SecurityFindingSeverity[] = [
  "CRITICAL",
  "HIGH",
  "ERROR",
  "WARNING",
  "MEDIUM",
  "LOW",
  "INFO",
  "EXPERIMENT",
  "INVENTORY",
];

export function severityRank(severity: SecurityFindingSeverity): number {
  const index = SECURITY_SEVERITY_ORDER.indexOf(severity);
  return index === -1 ? SECURITY_SEVERITY_ORDER.length : index;
}

export function severityBucket(
  severity: SecurityFindingSeverity,
): SecuritySeverityBucket {
  switch (severity) {
    case "CRITICAL":
    case "HIGH":
    case "ERROR":
      return "high";
    case "WARNING":
    case "MEDIUM":
      return "medium";
    case "LOW":
      return "low";
    default:
      return "info";
  }
}

export function toSastViewModel(
  finding: SecurityFinding,
): SecurityFindingViewModel {
  return {
    id: `sast:${finding.id}`,
    tool: "sast",
    severity: finding.severity,
    severityBucket: severityBucket(finding.severity),
    reference: finding.ruleId,
    description: finding.message,
    location: `${finding.filePath}:${finding.startLine}:${finding.startCol}`,
    filePath: finding.filePath,
  };
}

export function toScaViewModel(finding: ScaFinding): SecurityFindingViewModel {
  return {
    id: `sca:${finding.id}`,
    tool: "sca",
    severity: finding.severity,
    severityBucket: severityBucket(finding.severity),
    reference: finding.cveId,
    description: finding.description,
    location: `${finding.packageName}@${finding.packageVersion}`,
    packageName: finding.packageName,
  };
}

export function buildUnifiedSecurityFindings(
  sast: SecurityScanResult | null,
  sca: ScaScanResult | null,
): SecurityFindingViewModel[] {
  const rows: SecurityFindingViewModel[] = [];
  if (sast) {
    for (const finding of sast.findings) {
      rows.push(toSastViewModel(finding));
    }
  }
  if (sca) {
    for (const finding of sca.findings) {
      rows.push(toScaViewModel(finding));
    }
  }
  return rows.sort(
    (a, b) =>
      severityRank(a.severity) - severityRank(b.severity) ||
      a.tool.localeCompare(b.tool) ||
      a.location.localeCompare(b.location) ||
      a.reference.localeCompare(b.reference),
  );
}

function matchesQuery(row: SecurityFindingViewModel, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    row.description.toLowerCase().includes(q) ||
    row.reference.toLowerCase().includes(q) ||
    row.location.toLowerCase().includes(q) ||
    (row.filePath?.toLowerCase().includes(q) ?? false) ||
    (row.packageName?.toLowerCase().includes(q) ?? false) ||
    row.tool.toLowerCase().includes(q)
  );
}

export function filterSecurityFindings(
  rows: readonly SecurityFindingViewModel[],
  filters: SecurityFindingsFilters,
): SecurityFindingViewModel[] {
  return rows.filter((row) => {
    if (filters.tool !== "all" && row.tool !== filters.tool) return false;
    if (filters.severity !== "all" && row.severityBucket !== filters.severity) {
      return false;
    }
    return matchesQuery(row, filters.query);
  });
}
