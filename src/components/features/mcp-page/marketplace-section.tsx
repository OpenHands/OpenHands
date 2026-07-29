import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import {
  getMarketplaceEntriesByPopularity,
  getMcpMarketplaceCatalog,
  marketplaceEntryMatchesQuery,
} from "#/utils/mcp-marketplace-utils";
import { MarketplaceCard } from "./marketplace-card";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
} from "#/utils/extension-module-card-classes";

interface MarketplaceSectionProps {
  onSelect: (entry: MarketplaceEntry) => void;
  onAdd: (entry: MarketplaceEntry) => void;
  /** Empty string = no filter. */
  query?: string;
}

export function MarketplaceSection({
  onSelect,
  onAdd,
  query = "",
}: MarketplaceSectionProps) {
  const { t } = useTranslation("openhands");

  const visibleEntries = getMarketplaceEntriesByPopularity(
    getMcpMarketplaceCatalog(MCP_MARKETPLACE),
  ).filter((entry) => marketplaceEntryMatchesQuery(entry, query));

  return (
    <section
      data-testid="mcp-marketplace-section"
      className="flex flex-col gap-3"
    >
      <h2 className="text-base font-medium text-foreground">
        {t(I18nKey.MCP$LIBRARY_TITLE)}
      </h2>

      {visibleEntries.length === 0 ? (
        <div
          data-testid="mcp-marketplace-empty"
          className="rounded-xl border border-dashed border-[var(--oh-border)] p-6 text-center"
        >
          <p className="text-xs text-tertiary-light">
            {t(I18nKey.MCP$SEARCH_EMPTY)}
          </p>
        </div>
      ) : (
        <div className={extensionModuleCardGridContainerClassName}>
          <div
            data-testid="mcp-marketplace-grid"
            className={extensionModuleCardGridClassName}
          >
            {visibleEntries.map((entry) => (
              <MarketplaceCard
                key={entry.id}
                entry={entry}
                onClick={() => onSelect(entry)}
                onAdd={() => onAdd(entry)}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
