import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { PluginStatusFilter } from "./build-plugins-view-model";

interface PluginsToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  statusFilter: PluginStatusFilter;
  onStatusFilterChange: (filter: PluginStatusFilter) => void;
}

const STATUS_FILTERS: ReadonlyArray<{
  value: PluginStatusFilter;
  labelKey: I18nKey;
}> = [
  { value: "all", labelKey: I18nKey.SETTINGS$PLUGINS_FILTER_ALL },
  { value: "installed", labelKey: I18nKey.SETTINGS$PLUGINS_FILTER_INSTALLED },
  { value: "available", labelKey: I18nKey.SETTINGS$PLUGINS_FILTER_AVAILABLE },
  { value: "local", labelKey: I18nKey.SETTINGS$PLUGINS_FILTER_LOCAL },
];

export function PluginsToolbar({
  search,
  onSearchChange,
  statusFilter,
  onStatusFilterChange,
}: PluginsToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="plugins-toolbar" className="flex items-stretch gap-2">
      <div
        className={cn(
          "relative flex flex-1 min-w-0 items-center",
          "rounded-lg border border-[var(--oh-border)] bg-base-secondary",
          "focus-within:border-white/40 focus-within:ring-1 focus-within:ring-white/20",
          "transition-colors",
        )}
      >
        <Search
          className="ml-3 h-4 w-4 shrink-0 text-tertiary-alt"
          aria-hidden
        />
        <input
          data-testid="plugins-search-input"
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={t(I18nKey.SETTINGS$PLUGINS_SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.SETTINGS$PLUGINS_SEARCH_PLACEHOLDER)}
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
            className="mr-2 p-1 rounded text-tertiary-alt hover:text-white cursor-pointer"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
      </div>

      <div
        data-testid="plugins-status-filter"
        className="flex shrink-0 items-center gap-1 rounded-lg border border-[var(--oh-border)] bg-base-secondary p-1"
      >
        {STATUS_FILTERS.map((filter) => (
          <button
            key={filter.value}
            type="button"
            data-testid={`plugins-filter-${filter.value}`}
            aria-pressed={statusFilter === filter.value}
            onClick={() => onStatusFilterChange(filter.value)}
            className={cn(
              "cursor-pointer rounded-md px-3 py-1 text-xs transition-colors",
              statusFilter === filter.value
                ? "bg-surface-raised text-white"
                : "text-tertiary-alt hover:text-white",
            )}
          >
            {t(filter.labelKey)}
          </button>
        ))}
      </div>
    </div>
  );
}
