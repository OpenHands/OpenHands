/**
 * Findings filter toolbar.
 * @spec PROJETOSIN-188 — findings-filters
 */

import { useTranslation } from "react-i18next";
import {
  FINDING_SEVERITIES,
  FINDING_STATUSES,
  type FindingSeverity,
  type FindingStatus,
} from "#/api/pentest/findings-types";
import { I18nKey } from "#/i18n/declaration";

export interface FindingsFilterState {
  severities: FindingSeverity[];
  statuses: FindingStatus[];
  sourceTool: string;
  asset: string;
  titleQuery: string;
}

interface FindingsFiltersProps {
  value: FindingsFilterState;
  toolOptions: string[];
  onChange: (next: FindingsFilterState) => void;
  onClear: () => void;
  hasActiveFilters: boolean;
}

const SEVERITY_LABEL: Record<FindingSeverity, I18nKey> = {
  critical: I18nKey.FINDINGS$SEVERITY_CRITICAL,
  high: I18nKey.FINDINGS$SEVERITY_HIGH,
  medium: I18nKey.FINDINGS$SEVERITY_MEDIUM,
  low: I18nKey.FINDINGS$SEVERITY_LOW,
  info: I18nKey.FINDINGS$SEVERITY_INFO,
};

const STATUS_LABEL: Record<FindingStatus, I18nKey> = {
  new: I18nKey.FINDINGS$STATUS_NEW,
  triaging: I18nKey.FINDINGS$STATUS_TRIAGING,
  confirmed: I18nKey.FINDINGS$STATUS_CONFIRMED,
  false_positive: I18nKey.FINDINGS$STATUS_FALSE_POSITIVE,
  duplicate: I18nKey.FINDINGS$STATUS_DUPLICATE,
  risk_accepted: I18nKey.FINDINGS$STATUS_RISK_ACCEPTED,
};

function toggleInList<T extends string>(list: T[], item: T): T[] {
  return list.includes(item)
    ? list.filter((value) => value !== item)
    : [...list, item];
}

export function FindingsFilters({
  value,
  toolOptions,
  onChange,
  onClear,
  hasActiveFilters,
}: FindingsFiltersProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="findings-filters"
      className="flex flex-col gap-3 rounded-xl border border-[var(--oh-border)] p-3"
    >
      <div className="flex flex-wrap gap-3">
        <fieldset data-testid="findings-filter-severity" className="min-w-0">
          <legend className="mb-1 text-xs text-[var(--oh-text-tertiary)]">
            {t(I18nKey.FINDINGS$FILTER_SEVERITY)}
          </legend>
          <div className="flex flex-wrap gap-1.5">
            {FINDING_SEVERITIES.map((severity) => {
              const active = value.severities.includes(severity);
              return (
                <button
                  key={severity}
                  type="button"
                  aria-pressed={active}
                  className={
                    active
                      ? "rounded-md border border-[var(--oh-color-primary)] px-2 py-1 text-xs text-[color:var(--oh-color-primary)]"
                      : "rounded-md border border-[var(--oh-border)] px-2 py-1 text-xs text-[var(--oh-text-secondary)]"
                  }
                  onClick={() =>
                    onChange({
                      ...value,
                      severities: toggleInList(value.severities, severity),
                    })
                  }
                >
                  {t(SEVERITY_LABEL[severity])}
                </button>
              );
            })}
          </div>
        </fieldset>

        <fieldset data-testid="findings-filter-status" className="min-w-0">
          <legend className="mb-1 text-xs text-[var(--oh-text-tertiary)]">
            {t(I18nKey.FINDINGS$FILTER_STATUS)}
          </legend>
          <div className="flex flex-wrap gap-1.5">
            {FINDING_STATUSES.map((status) => {
              const active = value.statuses.includes(status);
              return (
                <button
                  key={status}
                  type="button"
                  aria-pressed={active}
                  className={
                    active
                      ? "rounded-md border border-[var(--oh-color-primary)] px-2 py-1 text-xs text-[color:var(--oh-color-primary)]"
                      : "rounded-md border border-[var(--oh-border)] px-2 py-1 text-xs text-[var(--oh-text-secondary)]"
                  }
                  onClick={() =>
                    onChange({
                      ...value,
                      statuses: toggleInList(value.statuses, status),
                    })
                  }
                >
                  {t(STATUS_LABEL[status])}
                </button>
              );
            })}
          </div>
        </fieldset>
      </div>

      <div className="flex flex-wrap gap-3">
        <label className="flex min-w-[10rem] flex-1 flex-col gap-1 text-xs text-[var(--oh-text-tertiary)]">
          {t(I18nKey.FINDINGS$FILTER_TOOL)}
          <select
            data-testid="findings-filter-tool"
            className="rounded-md border border-[var(--oh-border)] bg-base-secondary px-2 py-1.5 text-sm text-white"
            value={value.sourceTool}
            onChange={(event) =>
              onChange({ ...value, sourceTool: event.target.value })
            }
          >
            <option value="">{t(I18nKey.FINDINGS$FILTER_TOOL)}</option>
            {toolOptions.map((tool) => (
              <option key={tool} value={tool}>
                {tool}
              </option>
            ))}
          </select>
        </label>

        <label className="flex min-w-[10rem] flex-1 flex-col gap-1 text-xs text-[var(--oh-text-tertiary)]">
          {t(I18nKey.FINDINGS$FILTER_ASSET)}
          <input
            data-testid="findings-filter-asset"
            type="text"
            className="rounded-md border border-[var(--oh-border)] bg-base-secondary px-2 py-1.5 text-sm text-white"
            value={value.asset}
            onChange={(event) =>
              onChange({ ...value, asset: event.target.value })
            }
          />
        </label>

        <label className="flex min-w-[12rem] flex-[1.5] flex-col gap-1 text-xs text-[var(--oh-text-tertiary)]">
          {t(I18nKey.FINDINGS$SEARCH_TITLE_PLACEHOLDER)}
          <input
            data-testid="findings-search-title"
            type="search"
            placeholder={t(I18nKey.FINDINGS$SEARCH_TITLE_PLACEHOLDER)}
            className="rounded-md border border-[var(--oh-border)] bg-base-secondary px-2 py-1.5 text-sm text-white"
            value={value.titleQuery}
            onChange={(event) =>
              onChange({ ...value, titleQuery: event.target.value })
            }
          />
        </label>
      </div>

      {hasActiveFilters ? (
        <button
          type="button"
          data-testid="findings-clear-filters"
          className="self-start text-sm text-[var(--oh-color-primary)] underline-offset-2 hover:underline"
          onClick={onClear}
        >
          {t(I18nKey.FINDINGS$CLEAR_FILTERS)}
        </button>
      ) : null}
    </div>
  );
}

export const EMPTY_FINDINGS_FILTERS: FindingsFilterState = {
  severities: [],
  statuses: [],
  sourceTool: "",
  asset: "",
  titleQuery: "",
};
