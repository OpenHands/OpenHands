import type {
  ScaFinding,
  SecurityFindingSeverity,
} from "#/types/security-scan";

interface DependencyTrackComponent {
  name?: string;
  version?: string;
  purl?: string;
  group?: string;
}

interface DependencyTrackVulnerability {
  vulnId?: string;
  severity?: string;
  description?: string;
  title?: string;
}

export interface DependencyTrackFinding {
  uuid?: string;
  matrix?: string;
  component?: DependencyTrackComponent;
  vulnerability?: DependencyTrackVulnerability;
}

const VALID_SEVERITIES: ReadonlySet<SecurityFindingSeverity> = new Set([
  "CRITICAL",
  "HIGH",
  "ERROR",
  "WARNING",
  "MEDIUM",
  "LOW",
  "INFO",
  "EXPERIMENT",
  "INVENTORY",
]);

function normalizeSeverity(value: string | undefined): SecurityFindingSeverity {
  const upper = String(value ?? "INFO").toUpperCase();
  if (VALID_SEVERITIES.has(upper as SecurityFindingSeverity)) {
    return upper as SecurityFindingSeverity;
  }
  return "INFO";
}

function buildFindingId(
  finding: DependencyTrackFinding,
  index: number,
): string {
  return (
    finding.uuid ||
    finding.matrix ||
    `${finding.vulnerability?.vulnId ?? "finding"}:${finding.component?.purl ?? finding.component?.name ?? "unknown"}:${index}`
  );
}

export function mapDependencyTrackFindings(
  findings: DependencyTrackFinding[],
): ScaFinding[] {
  return findings.map((finding, index) => {
    const component = finding.component ?? {};
    const vulnerability = finding.vulnerability ?? {};
    const packageName =
      component.name?.trim() ||
      component.purl?.trim() ||
      component.group?.trim() ||
      "unknown";

    return {
      id: buildFindingId(finding, index),
      packageName,
      packageVersion: component.version?.trim() || "—",
      purl: component.purl?.trim() || packageName,
      cveId: vulnerability.vulnId?.trim() || "—",
      severity: normalizeSeverity(vulnerability.severity),
      description:
        vulnerability.description?.trim() ||
        vulnerability.title?.trim() ||
        vulnerability.vulnId?.trim() ||
        packageName,
    };
  });
}
