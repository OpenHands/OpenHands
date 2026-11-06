import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface MarketplaceCatalogToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
}

/** The search bar above the facet menu and cards, as on the Skills page. */
export function MarketplaceCatalogToolbar({
  search,
  onSearchChange,
}: MarketplaceCatalogToolbarProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="marketplace-catalog-toolbar"
      className={cn(
        "relative flex w-full min-w-0 items-center",
        "rounded-lg border border-border bg-base-secondary",
        "focus-within:border-contrast/40 focus-within:ring-1 focus-within:ring-contrast/20",
        "transition-colors",
      )}
    >
      <Search className="ml-3 h-4 w-4 shrink-0 text-tertiary-alt" aria-hidden />
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
  );
}
