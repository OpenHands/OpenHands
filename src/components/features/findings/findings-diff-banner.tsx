/**
 * Stats chips + scan-diff unavailable note.
 * @spec PROJETOSIN-188 — findings-diff-banner
 */

import { useTranslation } from "react-i18next";
import type { FindingStats } from "#/api/pentest/findings-types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface FindingsDiffBannerProps {
  stats: FindingStats | undefined;
  newOnly: boolean;
  onToggleNewOnly: () => void;
}

export function FindingsDiffBanner({
  stats,
  newOnly,
  onToggleNewOnly,
}: FindingsDiffBannerProps) {
  const { t } = useTranslation("openhands");
  const byStatus = stats?.by_status ?? {};
  const total = stats?.total ?? 0;
  const countNew = byStatus.new ?? 0;
  const countConfirmed = byStatus.confirmed ?? 0;
  const countFp = byStatus.false_positive ?? 0;

  return (
    <div
      data-testid="findings-stats-banner"
      className="flex flex-col gap-2"
      aria-label={t(I18nKey.FINDINGS$STATS_LABEL)}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-md border border-[var(--oh-border)] px-2.5 py-1 text-xs text-[var(--oh-text-secondary)]">
          {t(I18nKey.FINDINGS$COUNT_TOTAL, { count: total })}
        </span>
        <span className="rounded-md border border-[var(--oh-border)] px-2.5 py-1 text-xs text-[var(--oh-text-secondary)]">
          {t(I18nKey.FINDINGS$COUNT_NEW, { count: countNew })}
        </span>
        <span className="rounded-md border border-[var(--oh-border)] px-2.5 py-1 text-xs text-[var(--oh-text-secondary)]">
          {t(I18nKey.FINDINGS$COUNT_CONFIRMED, { count: countConfirmed })}
        </span>
        <span className="rounded-md border border-[var(--oh-border)] px-2.5 py-1 text-xs text-[var(--oh-text-secondary)]">
          {t(I18nKey.FINDINGS$COUNT_FP, { count: countFp })}
        </span>
        <button
          type="button"
          data-testid="findings-filter-new-only"
          aria-pressed={newOnly}
          onClick={onToggleNewOnly}
          className={cn(
            "rounded-md border px-2.5 py-1 text-xs transition-colors",
            newOnly
              ? "border-[var(--oh-color-primary)] bg-[rgba(255,200,80,0.12)] text-[color:var(--oh-color-primary)]"
              : "border-[var(--oh-border)] text-[var(--oh-text-secondary)] hover:bg-[var(--oh-surface-raised)]",
          )}
        >
          {t(I18nKey.FINDINGS$FILTER_NEW_ONLY)}
        </button>
      </div>
      <p
        data-testid="findings-diff-unavailable"
        className="text-xs text-[var(--oh-text-tertiary)]"
      >
        {t(I18nKey.FINDINGS$DIFF_UNAVAILABLE)}
      </p>
    </div>
  );
}
