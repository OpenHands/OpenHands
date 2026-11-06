import { I18nKey } from "#/i18n/declaration";
import type { CatalogKind, MarketplaceScope } from "#/types/marketplaces";

/** Label key per catalog kind, in the order the UI lists them. */
export const CATALOG_KIND_LABEL_KEYS: Record<CatalogKind, I18nKey> = {
  mcp: I18nKey.MARKETPLACES$KIND_MCP,
  skill: I18nKey.MARKETPLACES$KIND_SKILL,
  plugin: I18nKey.MARKETPLACES$KIND_PLUGIN,
  app: I18nKey.MARKETPLACES$KIND_APP,
  automation: I18nKey.MARKETPLACES$KIND_AUTOMATION,
};

export const MARKETPLACE_SCOPE_LABEL_KEYS: Record<MarketplaceScope, I18nKey> = {
  instance: I18nKey.MARKETPLACES$SCOPE_INSTANCE,
  org: I18nKey.MARKETPLACES$SCOPE_ORG,
  user: I18nKey.MARKETPLACES$SCOPE_USER,
};

export function marketplacePath(name: string): string {
  return `/marketplaces/${encodeURIComponent(name)}`;
}
