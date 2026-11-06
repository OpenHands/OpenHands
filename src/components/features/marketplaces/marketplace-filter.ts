import { I18nKey } from "#/i18n/declaration";
import type {
  CatalogEntry,
  CatalogKind,
  InstallState,
} from "#/types/marketplaces";

export type KindTab = CatalogKind | "all";
export type ContentFacet = "runs_code" | "no_code";
export type MarketplaceFacetGroupId = "status" | "content" | "tag";

export interface MarketplaceFilterState {
  query: string;
  statuses: Set<InstallState>;
  contents: Set<ContentFacet>;
  tags: Set<string>;
}

export const EMPTY_MARKETPLACE_FILTER: MarketplaceFilterState = {
  query: "",
  statuses: new Set(),
  contents: new Set(),
  tags: new Set(),
};

export interface MarketplaceFacetRow {
  value: string;
  /** Either a translation key or a literal label (tags are free text). */
  labelKey?: I18nKey;
  label?: string;
  count: number;
  checked: boolean;
}

export interface MarketplaceFacetGroup {
  id: MarketplaceFacetGroupId;
  labelKey: I18nKey;
  rows: MarketplaceFacetRow[];
}

const STATUS_ORDER: readonly InstallState[] = [
  "installed",
  "update_available",
  "not_installed",
];

const STATUS_LABEL_KEYS: Record<InstallState, I18nKey> = {
  installed: I18nKey.MARKETPLACES$STATUS_INSTALLED,
  update_available: I18nKey.MARKETPLACES$STATUS_UPDATE_AVAILABLE,
  not_installed: I18nKey.MARKETPLACES$STATUS_NOT_INSTALLED,
};

const CONTENT_LABEL_KEYS: Record<ContentFacet, I18nKey> = {
  runs_code: I18nKey.MARKETPLACES$RUNS_CODE,
  no_code: I18nKey.MARKETPLACES$CONTENT_NO_CODE,
};

/** How many tags the rail lists; the rest stay searchable. */
const MAX_TAG_ROWS = 12;

function contentOf(entry: CatalogEntry): ContentFacet {
  return entry.runs_code ? "runs_code" : "no_code";
}

/** An entry with an update available is still installed, so it matches both. */
function statusesOf(entry: CatalogEntry): InstallState[] {
  const { state } = entry.install;
  return state === "update_available" ? ["installed", state] : [state];
}

export function matchesQuery(entry: CatalogEntry, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [entry.name, entry.description, ...entry.tags]
    .join(" ")
    .toLowerCase()
    .includes(needle);
}

export function matchesTab(entry: CatalogEntry, tab: KindTab): boolean {
  return tab === "all" || entry.kind === tab;
}

/** Facets OR within a group and AND across groups, as the Skills page does. */
export function matchesFacets(
  entry: CatalogEntry,
  filter: MarketplaceFilterState,
): boolean {
  if (
    filter.statuses.size > 0 &&
    !statusesOf(entry).some((status) => filter.statuses.has(status))
  ) {
    return false;
  }
  if (filter.contents.size > 0 && !filter.contents.has(contentOf(entry))) {
    return false;
  }
  if (filter.tags.size > 0 && !entry.tags.some((tag) => filter.tags.has(tag))) {
    return false;
  }
  return true;
}

export function filterEntries(
  entries: CatalogEntry[],
  tab: KindTab,
  filter: MarketplaceFilterState,
): CatalogEntry[] {
  return entries.filter(
    (entry) =>
      matchesTab(entry, tab) &&
      matchesQuery(entry, filter.query) &&
      matchesFacets(entry, filter),
  );
}

export function countActiveFacets(filter: MarketplaceFilterState): number {
  return filter.statuses.size + filter.contents.size + filter.tags.size;
}

/**
 * Facet rows for the current tab and search. Counts ignore the facets
 * themselves so a row still says how many entries it would add.
 */
export function buildFacetGroups(
  entries: CatalogEntry[],
  tab: KindTab,
  filter: MarketplaceFilterState,
): MarketplaceFacetGroup[] {
  const scoped = entries.filter(
    (entry) => matchesTab(entry, tab) && matchesQuery(entry, filter.query),
  );
  const count = <T>(values: T[], pick: (entry: CatalogEntry) => T) =>
    values.map((value) => scoped.filter((e) => pick(e) === value).length);

  const statusCounts = STATUS_ORDER.map(
    (status) => scoped.filter((e) => statusesOf(e).includes(status)).length,
  );
  const contentValues: ContentFacet[] = ["runs_code", "no_code"];
  const contentCounts = count(contentValues, contentOf);

  const tagCounts = new Map<string, number>();
  for (const entry of scoped) {
    for (const tag of entry.tags) {
      tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
    }
  }
  const topTags = Array.from(tagCounts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, MAX_TAG_ROWS);
  const shown = new Set(topTags.map(([tag]) => tag));
  // A selected tag stays visible even when it drops out of the top rows.
  const selectedExtra = Array.from(filter.tags)
    .filter((tag) => !shown.has(tag))
    .map((tag) => [tag, tagCounts.get(tag) ?? 0] as [string, number]);
  const tagRows = [...topTags, ...selectedExtra].map(([tag, n]) => ({
    value: tag,
    label: tag,
    count: n,
    checked: filter.tags.has(tag),
  }));

  const groups: MarketplaceFacetGroup[] = [
    {
      id: "status",
      labelKey: I18nKey.MARKETPLACES$FACET_STATUS,
      rows: STATUS_ORDER.map((status, i) => ({
        value: status,
        labelKey: STATUS_LABEL_KEYS[status],
        count: statusCounts[i],
        checked: filter.statuses.has(status),
      })),
    },
    {
      id: "content",
      labelKey: I18nKey.MARKETPLACES$FACET_CONTENT,
      rows: contentValues.map((value, i) => ({
        value,
        labelKey: CONTENT_LABEL_KEYS[value],
        count: contentCounts[i],
        checked: filter.contents.has(value),
      })),
    },
  ];
  if (tagRows.length > 0) {
    groups.push({
      id: "tag",
      labelKey: I18nKey.MARKETPLACES$FACET_TAGS,
      rows: tagRows,
    });
  }
  return groups;
}

export function toggleFacet(
  filter: MarketplaceFilterState,
  groupId: MarketplaceFacetGroupId,
  value: string,
): MarketplaceFilterState {
  const toggle = <T>(set: Set<T>, item: T): Set<T> => {
    const next = new Set(set);
    if (next.has(item)) next.delete(item);
    else next.add(item);
    return next;
  };
  switch (groupId) {
    case "status":
      return {
        ...filter,
        statuses: toggle(filter.statuses, value as InstallState),
      };
    case "content":
      return {
        ...filter,
        contents: toggle(filter.contents, value as ContentFacet),
      };
    case "tag":
      return { ...filter, tags: toggle(filter.tags, value) };
    default:
      return filter;
  }
}
