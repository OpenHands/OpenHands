import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { McpSectionFilter } from "./mcp-section-filter";
import { McpSectionFilterDropdown } from "./mcp-section-filter-dropdown";

interface McpToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  sectionFilter: McpSectionFilter;
  onSectionFilterChange: (filter: McpSectionFilter) => void;
}

/**
 * Full-width search plus section filter for the MCP page. Filters both the
 * Installed and Library sections (or limits which section is visible).
 */
export function McpToolbar({
  search,
  onSearchChange,
  sectionFilter,
  onSectionFilterChange,
}: McpToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="mcp-toolbar" className="flex items-stretch gap-2">
      <div
        data-testid="mcp-search"
        className={cn(
          "relative flex flex-1 min-w-0 items-center",
          "rounded-lg border border-[var(--oh-border)] bg-base-secondary",
          "focus-within:border-white/40 focus-within:ring-1 focus-within:ring-white/20",
          "transition-colors",
        )}
      >
        <Search
          className="ml-3 h-4 w-4 text-tertiary-alt shrink-0"
          aria-hidden
        />
        <input
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={t(I18nKey.MCP$SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.MCP$SEARCH_PLACEHOLDER)}
          data-testid="mcp-search-input"
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
            data-testid="mcp-search-clear"
            className="mr-2 p-1 rounded text-tertiary-alt hover:text-content-1 cursor-pointer"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
      </div>

      <McpSectionFilterDropdown
        value={sectionFilter}
        onChange={onSectionFilterChange}
      />
    </div>
  );
}
