import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { CATALOG_KINDS, type CatalogKind } from "#/types/marketplaces";
import { CATALOG_KIND_LABEL_KEYS } from "./catalog-kinds";

export type KindFilter = CatalogKind | "all";
export type InstallFilter = "all" | "installed";

interface MarketplaceCatalogToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  kind: KindFilter;
  onKindChange: (kind: KindFilter) => void;
  counts: Record<CatalogKind, number>;
  installFilter: InstallFilter;
  onInstallFilterChange: (filter: InstallFilter) => void;
  installedCount: number;
}

const segmentClassName = (active: boolean) =>
  cn(
    "cursor-pointer rounded-md px-3 py-1 text-xs transition-colors whitespace-nowrap",
    active
      ? "bg-surface-raised text-contrast"
      : "text-tertiary-alt hover:text-contrast",
  );

export function MarketplaceCatalogToolbar({
  search,
  onSearchChange,
  kind,
  onKindChange,
  counts,
  installFilter,
  onInstallFilterChange,
  installedCount,
}: MarketplaceCatalogToolbarProps) {
  const { t } = useTranslation("openhands");
  const total = CATALOG_KINDS.reduce((sum, k) => sum + counts[k], 0);

  return (
    <div
      data-testid="marketplace-catalog-toolbar"
      className="flex flex-col gap-2"
    >
      <div className="flex flex-col items-stretch gap-2 md:flex-row">
        <div
          className={cn(
            "relative flex w-full min-w-0 flex-1 items-center",
            "rounded-lg border border-border bg-base-secondary",
            "focus-within:border-contrast/40 focus-within:ring-1 focus-within:ring-contrast/20",
            "transition-colors",
          )}
        >
          <Search
            className="ml-3 h-4 w-4 shrink-0 text-tertiary-alt"
            aria-hidden
          />
          <input
            data-testid="marketplace-search-input"
            type="search"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder={t(I18nKey.MARKETPLACES$SEARCH_PLACEHOLDER)}
            aria-label={t(I18nKey.MARKETPLACES$SEARCH_PLACEHOLDER)}
            className={cn(
              "flex-1 min-w-0 bg-transparent border-0 outline-none",
              "px-3 py-2 text-sm placeholder:text-tertiary-alt",
              "[&::-webkit-search-cancel-button]:hidden",
            )}
          />
          {search ? (
            <button
              type="button"
              onClick={() => onSearchChange("")}
              aria-label={t(I18nKey.MCP$SEARCH_CLEAR)}
              className="mr-2 p-1 rounded text-tertiary-alt hover:text-contrast cursor-pointer"
            >
              <X className="h-4 w-4" aria-hidden />
            </button>
          ) : null}
        </div>

        <div
          data-testid="marketplace-install-filter"
          className="flex w-full items-center gap-1 rounded-lg border border-border bg-base-secondary p-1 md:w-auto md:shrink-0"
        >
          <button
            type="button"
            data-testid="marketplace-install-filter-all"
            aria-pressed={installFilter === "all"}
            onClick={() => onInstallFilterChange("all")}
            className={segmentClassName(installFilter === "all")}
          >
            {t(I18nKey.MARKETPLACES$FILTER_EVERYTHING)}
          </button>
          <button
            type="button"
            data-testid="marketplace-install-filter-installed"
            aria-pressed={installFilter === "installed"}
            onClick={() => onInstallFilterChange("installed")}
            className={segmentClassName(installFilter === "installed")}
          >
            {t(I18nKey.MARKETPLACES$FILTER_INSTALLED, {
              count: installedCount,
            })}
          </button>
        </div>
      </div>

      <div
        data-testid="marketplace-kind-filter"
        role="group"
        aria-label={t(I18nKey.MARKETPLACES$KIND_FILTER_LABEL)}
        className="flex w-full flex-wrap items-center gap-1 rounded-lg border border-border bg-base-secondary p-1"
      >
        <button
          type="button"
          data-testid="marketplace-kind-all"
          aria-pressed={kind === "all"}
          onClick={() => onKindChange("all")}
          className={segmentClassName(kind === "all")}
        >
          {t(I18nKey.MARKETPLACES$KIND_ALL)} ({total})
        </button>
        {CATALOG_KINDS.map((value) => (
          <button
            key={value}
            type="button"
            data-testid={`marketplace-kind-${value}`}
            aria-pressed={kind === value}
            disabled={counts[value] === 0}
            onClick={() => onKindChange(value)}
            className={cn(
              segmentClassName(kind === value),
              counts[value] === 0 && "cursor-not-allowed opacity-40",
            )}
          >
            {t(CATALOG_KIND_LABEL_KEYS[value])} ({counts[value]})
          </button>
        ))}
      </div>
    </div>
  );
}
