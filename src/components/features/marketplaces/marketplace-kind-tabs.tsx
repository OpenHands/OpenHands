import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { TabButton } from "#/components/features/conversation-panel/system-message-modal/tab-button";
import { CATALOG_KINDS, type CatalogKind } from "#/types/marketplaces";
import { CATALOG_KIND_LABEL_KEYS } from "./catalog-kinds";
import type { KindTab } from "./marketplace-filter";

interface MarketplaceKindTabsProps {
  active: KindTab;
  counts: Record<CatalogKind, number>;
  onChange: (tab: KindTab) => void;
}

/** One tab per kind the marketplace ships, plus everything together. */
export function MarketplaceKindTabs({
  active,
  counts,
  onChange,
}: MarketplaceKindTabsProps) {
  const { t } = useTranslation("openhands");
  const total = CATALOG_KINDS.reduce((sum, kind) => sum + counts[kind], 0);

  return (
    <div
      data-testid="marketplace-kind-tabs"
      role="tablist"
      aria-label={t(I18nKey.MARKETPLACES$KIND_FILTER_LABEL)}
      className="flex min-w-0 overflow-x-auto border-b border-border text-sm"
    >
      <TabButton
        isActive={active === "all"}
        onClick={() => onChange("all")}
        className="shrink-0"
      >
        <span data-testid="marketplace-tab-all">
          {t(I18nKey.MARKETPLACES$KIND_ALL)} ({total})
        </span>
      </TabButton>
      {CATALOG_KINDS.map((kind) => (
        <TabButton
          key={kind}
          isActive={active === kind}
          disabled={counts[kind] === 0}
          onClick={() => onChange(kind)}
          className="shrink-0"
        >
          <span data-testid={`marketplace-tab-${kind}`}>
            {t(CATALOG_KIND_LABEL_KEYS[kind])} ({counts[kind]})
          </span>
        </TabButton>
      ))}
    </div>
  );
}
