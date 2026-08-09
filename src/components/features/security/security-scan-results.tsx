import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, FileWarning, Info, ShieldAlert } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type {
  SecurityFinding,
  SecurityFindingSeverity,
  SecurityScanResult,
} from "#/types/security-scan";
import { cn } from "#/utils/utils";

interface SecurityScanResultsProps {
  result: SecurityScanResult | null;
}

const SEVERITY_ORDER: SecurityFindingSeverity[] = [
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

function severityRank(severity: SecurityFindingSeverity): number {
  const index = SEVERITY_ORDER.indexOf(severity);
  return index === -1 ? SEVERITY_ORDER.length : index;
}

function SeverityIcon({
  severity,
}: {
  severity: SecurityFindingSeverity;
}) {
  if (severity === "CRITICAL" || severity === "HIGH" || severity === "ERROR") {
    return <ShieldAlert className="h-4 w-4 shrink-0 text-red-400" aria-hidden />;
  }
  if (severity === "WARNING" || severity === "MEDIUM") {
    return (
      <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" aria-hidden />
    );
  }
  return <Info className="h-4 w-4 shrink-0 text-sky-400" aria-hidden />;
}

function severityLabelKey(
  severity: SecurityFindingSeverity,
): I18nKey {
  switch (severity) {
    case "CRITICAL":
    case "HIGH":
    case "ERROR":
      return I18nKey.SECURITY$SEVERITY_HIGH;
    case "WARNING":
    case "MEDIUM":
      return I18nKey.SECURITY$SEVERITY_MEDIUM;
    case "LOW":
      return I18nKey.SECURITY$SEVERITY_LOW;
    default:
      return I18nKey.SECURITY$SEVERITY_INFO;
  }
}

function severityBadgeClass(severity: SecurityFindingSeverity): string {
  switch (severity) {
    case "CRITICAL":
    case "HIGH":
    case "ERROR":
      return "bg-red-500/15 text-red-300";
    case "WARNING":
    case "MEDIUM":
      return "bg-amber-500/15 text-amber-300";
    case "LOW":
      return "bg-sky-500/15 text-sky-300";
    default:
      return "bg-[var(--oh-surface-raised)] text-[var(--oh-muted)]";
  }
}

function SecurityFindingRow({ finding }: { finding: SecurityFinding }) {
  const { t } = useTranslation("openhands");

  return (
    <li
      className="rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3"
      data-testid={`security-finding-${finding.id}`}
    >
      <div className="flex items-start gap-2">
        <SeverityIcon severity={finding.severity} />
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                severityBadgeClass(finding.severity),
              )}
            >
              {t(severityLabelKey(finding.severity))}
            </span>
            <span className="truncate font-mono text-[11px] text-[var(--oh-muted)]">
              {finding.ruleId}
            </span>
          </div>
          <p className="text-sm text-white">{finding.message}</p>
          <p className="mt-2 font-mono text-xs text-[var(--oh-muted)]">
            {finding.filePath}:{finding.startLine}:{finding.startCol}
          </p>
        </div>
      </div>
    </li>
  );
}

export function SecurityScanResults({ result }: SecurityScanResultsProps) {
  const { t } = useTranslation("openhands");

  const sortedFindings = useMemo(() => {
    if (!result) return [];
    return [...result.findings].sort(
      (a, b) =>
        severityRank(a.severity) - severityRank(b.severity) ||
        a.filePath.localeCompare(b.filePath) ||
        a.startLine - b.startLine,
    );
  }, [result]);

  if (!result) {
    return (
      <div
        className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center"
        data-testid="security-empty-state"
      >
        <FileWarning className="h-8 w-8 text-[var(--oh-muted)]" aria-hidden />
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.SECURITY$EMPTY_STATE)}
        </p>
      </div>
    );
  }

  if (sortedFindings.length === 0) {
    return (
      <div
        className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center"
        data-testid="security-no-findings"
      >
        <ShieldAlert className="h-8 w-8 text-emerald-400" aria-hidden />
        <p className="text-sm text-white">{t(I18nKey.SECURITY$NO_FINDINGS)}</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="security-results">
      <p className="mb-3 text-xs text-[var(--oh-muted)]">
        {t(I18nKey.SECURITY$FINDINGS_COUNT, { count: sortedFindings.length })}
      </p>
      <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto">
        {sortedFindings.map((finding) => (
          <SecurityFindingRow key={finding.id} finding={finding} />
        ))}
      </ul>
    </div>
  );
}
