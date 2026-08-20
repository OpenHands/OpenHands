import { ListFilter, Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface SkillsToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  activeFilterCount: number;
  onOpenFilters: () => void;
}

export function SkillsToolbar({
  search,
  onSearchChange,
  activeFilterCount,
  onOpenFilters,
}: SkillsToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="skills-toolbar" className="flex items-stretch gap-2">
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
          data-testid="skills-search-input"
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={t(I18nKey.SETTINGS$SKILLS_SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.SETTINGS$SKILLS_SEARCH_PLACEHOLDER)}
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

      {/* The facet rail is desktop-only, so below md this is the only way in. */}
      <button
        type="button"
        data-testid="skills-filters-button"
        onClick={onOpenFilters}
        className={cn(
          "flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm md:hidden",
          "border border-[var(--oh-border)] bg-base-secondary text-white",
          "cursor-pointer hover:bg-[var(--oh-interactive-hover)]",
        )}
      >
        <ListFilter className="size-4 shrink-0" aria-hidden />
        {t(I18nKey.SETTINGS$SKILLS_FILTERS_BUTTON)}
        {activeFilterCount > 0 ? (
          <span className="rounded-full bg-white px-1.5 text-[11px] font-medium text-black">
            {activeFilterCount}
          </span>
        ) : null}
      </button>
    </div>
  );
}
