import { useTranslation } from "react-i18next";
import { Download, Search } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type {
  SecurityFindingsExportFormat,
} from "#/utils/security-findings-export";
import type {
  SecurityFindingsFilters,
  SecuritySeverityFilter,
  SecurityToolFilter,
} from "#/utils/security-findings-view";
import { cn } from "#/utils/utils";

interface SecurityFindingsToolbarProps {
  filters: SecurityFindingsFilters;
  onFiltersChange: (next: SecurityFindingsFilters) => void;
  shownCount: number;
  totalCount: number;
  canExport: boolean;
  onExport: (format: SecurityFindingsExportFormat) => void;
}

const selectClassName = cn(
  "h-8 rounded border border-[var(--oh-border)] bg-[var(--oh-surface)]",
  "px-2 text-xs text-white outline-none",
  "focus-visible:ring-1 focus-visible:ring-white/40",
);

const buttonClassName = cn(
  "flex h-8 items-center gap-1.5 rounded border border-[var(--oh-border)]",
  "bg-[var(--oh-surface)] px-2.5 text-xs text-white transition-opacity",
  "hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50",
);

export function SecurityFindingsToolbar({
  filters,
  onFiltersChange,
  shownCount,
  totalCount,
  canExport,
  onExport,
}: SecurityFindingsToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className="mb-3 flex flex-col gap-2"
      data-testid="security-findings-toolbar"
    >
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 text-xs text-[var(--oh-muted)]">
          <span className="sr-only">{t(I18nKey.SECURITY$FILTER_TOOL)}</span>
          <select
            data-testid="security-filter-tool"
            aria-label={t(I18nKey.SECURITY$FILTER_TOOL)}
            className={selectClassName}
            value={filters.tool}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                tool: event.target.value as SecurityToolFilter,
              })
            }
          >
            <option value="all">{t(I18nKey.SECURITY$FILTER_TOOL_ALL)}</option>
            <option value="sast">{t(I18nKey.SECURITY$FILTER_TOOL_SAST)}</option>
            <option value="sca">{t(I18nKey.SECURITY$FILTER_TOOL_SCA)}</option>
          </select>
        </label>

        <label className="flex items-center gap-1.5 text-xs text-[var(--oh-muted)]">
          <span className="sr-only">{t(I18nKey.SECURITY$FILTER_SEVERITY)}</span>
          <select
            data-testid="security-filter-severity"
            aria-label={t(I18nKey.SECURITY$FILTER_SEVERITY)}
            className={selectClassName}
            value={filters.severity}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                severity: event.target.value as SecuritySeverityFilter,
              })
            }
          >
            <option value="all">
              {t(I18nKey.SECURITY$FILTER_SEVERITY_ALL)}
            </option>
            <option value="high">{t(I18nKey.SECURITY$SEVERITY_HIGH)}</option>
            <option value="medium">
              {t(I18nKey.SECURITY$SEVERITY_MEDIUM)}
            </option>
            <option value="low">{t(I18nKey.SECURITY$SEVERITY_LOW)}</option>
            <option value="info">{t(I18nKey.SECURITY$SEVERITY_INFO)}</option>
          </select>
        </label>

        <label className="relative min-w-[180px] flex-1">
          <Search
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--oh-muted)]"
            aria-hidden
          />
          <span className="sr-only">{t(I18nKey.SECURITY$FILTER_SEARCH)}</span>
          <input
            data-testid="security-filter-search"
            type="search"
            aria-label={t(I18nKey.SECURITY$FILTER_SEARCH)}
            placeholder={t(I18nKey.SECURITY$FILTER_SEARCH_PLACEHOLDER)}
            value={filters.query}
            onChange={(event) =>
              onFiltersChange({ ...filters, query: event.target.value })
            }
            className={cn(
              selectClassName,
              "w-full pl-7 pr-2 placeholder:text-[var(--oh-muted)]",
            )}
          />
        </label>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="sr-only">{t(I18nKey.SECURITY$EXPORT)}</span>
          <button
            type="button"
            data-testid="security-export-csv"
            className={buttonClassName}
            disabled={!canExport}
            onClick={() => onExport("csv")}
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            {t(I18nKey.SECURITY$EXPORT_CSV)}
          </button>
          <button
            type="button"
            data-testid="security-export-excel"
            className={buttonClassName}
            disabled={!canExport}
            onClick={() => onExport("excel")}
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            {t(I18nKey.SECURITY$EXPORT_EXCEL)}
          </button>
          <button
            type="button"
            data-testid="security-export-pdf"
            className={buttonClassName}
            disabled={!canExport}
            onClick={() => onExport("pdf")}
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            {t(I18nKey.SECURITY$EXPORT_PDF)}
          </button>
        </div>
      </div>

      {totalCount > 0 && (
        <p
          className="text-xs text-[var(--oh-muted)]"
          data-testid="security-filtered-count"
        >
          {t(I18nKey.SECURITY$FILTERED_COUNT, {
            shown: shownCount,
            total: totalCount,
          })}
        </p>
      )}
    </div>
  );
}
