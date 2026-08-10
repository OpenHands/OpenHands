import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  FileWarning,
  Info,
  Package,
  ShieldAlert,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { SecurityFindingsToolbar } from "#/components/features/security/security-findings-toolbar";
import {
  displayTranslatedText,
  useTranslatedTexts,
} from "#/hooks/use-translated-texts";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";
import {
  exportSecurityFindings,
  type SecurityFindingsExportFormat,
  type SecurityFindingsExportLabels,
} from "#/utils/security-findings-export";
import {
  buildUnifiedSecurityFindings,
  DEFAULT_SECURITY_FINDINGS_FILTERS,
  filterSecurityFindings,
  type SecurityFindingViewModel,
  type SecurityFindingsFilters,
} from "#/utils/security-findings-view";
import { cn } from "#/utils/utils";

interface SecurityFindingsPanelProps {
  sastResult: SecurityScanResult | null;
  scaResult: ScaScanResult | null;
}

function SeverityIcon({
  bucket,
}: {
  bucket: SecurityFindingViewModel["severityBucket"];
}) {
  if (bucket === "high") {
    return (
      <ShieldAlert className="h-4 w-4 shrink-0 text-red-400" aria-hidden />
    );
  }
  if (bucket === "medium") {
    return (
      <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" aria-hidden />
    );
  }
  if (bucket === "low") {
    return <Info className="h-4 w-4 shrink-0 text-sky-400" aria-hidden />;
  }
  return (
    <Info className="h-4 w-4 shrink-0 text-[var(--oh-muted)]" aria-hidden />
  );
}

function severityLabelKey(
  bucket: SecurityFindingViewModel["severityBucket"],
): I18nKey {
  switch (bucket) {
    case "high":
      return I18nKey.SECURITY$SEVERITY_HIGH;
    case "medium":
      return I18nKey.SECURITY$SEVERITY_MEDIUM;
    case "low":
      return I18nKey.SECURITY$SEVERITY_LOW;
    default:
      return I18nKey.SECURITY$SEVERITY_INFO;
  }
}

function severityBadgeClass(
  bucket: SecurityFindingViewModel["severityBucket"],
): string {
  switch (bucket) {
    case "high":
      return "bg-red-500/15 text-red-300";
    case "medium":
      return "bg-amber-500/15 text-amber-300";
    case "low":
      return "bg-sky-500/15 text-sky-300";
    default:
      return "bg-[var(--oh-surface-raised)] text-[var(--oh-muted)]";
  }
}

function FindingRow({
  finding,
  description,
}: {
  finding: SecurityFindingViewModel;
  description: string;
}) {
  const { t } = useTranslation("openhands");
  const toolLabel =
    finding.tool === "sast"
      ? t(I18nKey.SECURITY$FILTER_TOOL_SAST)
      : t(I18nKey.SECURITY$FILTER_TOOL_SCA);
  const ToolIcon = finding.tool === "sast" ? FileWarning : Package;

  return (
    <li
      className="rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3"
      data-testid={`security-finding-${finding.id}`}
    >
      <div className="flex items-start gap-2">
        <SeverityIcon bucket={finding.severityBucket} />
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                severityBadgeClass(finding.severityBucket),
              )}
            >
              {t(severityLabelKey(finding.severityBucket))}
            </span>
            <span className="inline-flex items-center gap-1 rounded bg-[var(--oh-surface-raised)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--oh-muted)]">
              <ToolIcon className="h-3 w-3" aria-hidden />
              {toolLabel}
            </span>
            <span className="truncate font-mono text-[11px] text-[var(--oh-muted)]">
              {finding.reference}
            </span>
          </div>
          <p className="text-sm text-white">{description}</p>
          <p className="mt-2 font-mono text-xs text-[var(--oh-muted)]">
            {finding.location}
          </p>
        </div>
      </div>
    </li>
  );
}

export function SecurityFindingsPanel({
  sastResult,
  scaResult,
}: SecurityFindingsPanelProps) {
  const { t } = useTranslation("openhands");
  const [filters, setFilters] = useState<SecurityFindingsFilters>(
    DEFAULT_SECURITY_FINDINGS_FILTERS,
  );

  const allFindings = useMemo(
    () => buildUnifiedSecurityFindings(sastResult, scaResult),
    [sastResult, scaResult],
  );

  const filteredFindings = useMemo(
    () => filterSecurityFindings(allFindings, filters),
    [allFindings, filters],
  );

  const descriptions = useMemo(
    () => filteredFindings.map((finding) => finding.description),
    [filteredFindings],
  );
  const { translations } = useTranslatedTexts(descriptions);

  const exportLabels = useMemo<SecurityFindingsExportLabels>(
    () => ({
      title: t(I18nKey.COMMON$SECURITY),
      tool: t(I18nKey.SECURITY$FILTER_TOOL),
      severity: t(I18nKey.SECURITY$FILTER_SEVERITY),
      reference: t(I18nKey.SECURITY$EXPORT_COLUMN_REFERENCE),
      description: t(I18nKey.SECURITY$EXPORT_COLUMN_DESCRIPTION),
      location: t(I18nKey.SECURITY$EXPORT_COLUMN_LOCATION),
      toolSast: t(I18nKey.SECURITY$FILTER_TOOL_SAST),
      toolSca: t(I18nKey.SECURITY$FILTER_TOOL_SCA),
      severityHigh: t(I18nKey.SECURITY$SEVERITY_HIGH),
      severityMedium: t(I18nKey.SECURITY$SEVERITY_MEDIUM),
      severityLow: t(I18nKey.SECURITY$SEVERITY_LOW),
      severityInfo: t(I18nKey.SECURITY$SEVERITY_INFO),
    }),
    [t],
  );

  const handleExport = (format: SecurityFindingsExportFormat) => {
    if (filteredFindings.length === 0) return;
    const rows = filteredFindings.map((finding) => ({
      ...finding,
      description: displayTranslatedText(finding.description, translations),
    }));
    exportSecurityFindings(rows, format, exportLabels);
  };

  const hasAnyScan = sastResult != null || scaResult != null;

  if (!hasAnyScan) {
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

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      data-testid="security-findings-panel"
    >
      <SecurityFindingsToolbar
        filters={filters}
        onFiltersChange={setFilters}
        shownCount={filteredFindings.length}
        totalCount={allFindings.length}
        canExport={filteredFindings.length > 0}
        onExport={handleExport}
      />

      {allFindings.length === 0 ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
          data-testid="security-no-findings"
        >
          <ShieldAlert className="h-8 w-8 text-emerald-400" aria-hidden />
          <p className="text-sm text-white">
            {t(I18nKey.SECURITY$NO_FINDINGS)}
          </p>
        </div>
      ) : filteredFindings.length === 0 ? (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
          data-testid="security-no-filtered-findings"
        >
          <SearchEmptyIcon />
          <p className="text-sm text-white">
            {t(I18nKey.SECURITY$NO_FILTERED_FINDINGS)}
          </p>
        </div>
      ) : (
        <ul
          className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto"
          data-testid="security-results"
        >
          {filteredFindings.map((finding) => (
            <FindingRow
              key={finding.id}
              finding={finding}
              description={displayTranslatedText(
                finding.description,
                translations,
              )}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function SearchEmptyIcon() {
  return <Info className="h-8 w-8 text-[var(--oh-muted)]" aria-hidden />;
}
