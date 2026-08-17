import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { AgentLevelFilter } from "./agent-level-filter";
import { AgentsLevelFilterDropdown } from "./agents-level-filter-dropdown";

interface AgentsToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  levelFilter: AgentLevelFilter;
  onLevelFilterChange: (filter: AgentLevelFilter) => void;
}

export function AgentsToolbar({
  search,
  onSearchChange,
  levelFilter,
  onLevelFilterChange,
}: AgentsToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="agents-toolbar" className="flex items-stretch gap-2">
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
          data-testid="agents-search-input"
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={t(I18nKey.SETTINGS$AGENTS_SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.SETTINGS$AGENTS_SEARCH_PLACEHOLDER)}
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

      <AgentsLevelFilterDropdown
        value={levelFilter}
        onChange={onLevelFilterChange}
      />
    </div>
  );
}
