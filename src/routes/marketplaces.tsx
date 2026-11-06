import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { AddMarketplaceModal } from "#/components/features/marketplaces/add-marketplace-modal";
import { CatalogEntryCard } from "#/components/features/marketplaces/catalog-entry-card";
import { CatalogErrors } from "#/components/features/marketplaces/catalog-errors";
import { marketplacePath } from "#/components/features/marketplaces/catalog-kinds";
import { MarketplaceCatalogToolbar } from "#/components/features/marketplaces/marketplace-catalog-toolbar";
import { MarketplaceFacetRail } from "#/components/features/marketplaces/marketplace-facet-rail";
import {
  buildFacetGroups,
  countActiveFacets,
  EMPTY_MARKETPLACE_FILTER,
  filterEntries,
  toggleFacet,
  type KindTab,
  type MarketplaceFilterState,
} from "#/components/features/marketplaces/marketplace-filter";
import { MarketplaceKindTabs } from "#/components/features/marketplaces/marketplace-kind-tabs";
import { MarketplacePickerBar } from "#/components/features/marketplaces/marketplace-picker-bar";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useMarketplaces } from "#/hooks/query/use-marketplaces";
import { useMarketplaceCatalog } from "#/hooks/query/use-marketplace-catalog";
import {
  useRefreshMarketplace,
  useRemoveMarketplace,
  useUpdateMarketplace,
} from "#/hooks/mutation/use-manage-marketplaces";
import {
  useManageMarketplaceEntry,
  type EntryAction,
} from "#/hooks/mutation/use-manage-marketplace-entries";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { CatalogEntry, CatalogKind } from "#/types/marketplaces";

const MARKETPLACES_PATH = "/marketplaces";

const EMPTY_COUNTS: Record<CatalogKind, number> = {
  mcp: 0,
  skill: 0,
  plugin: 0,
  app: 0,
  automation: 0,
};

/**
 * One page for every marketplace: pick one at the top, then browse everything
 * it ships by kind. `/marketplaces/:marketplaceName` selects one; the bare
 * path selects the default.
 */
export default function MarketplacesScreen() {
  const { t } = useTranslation("openhands");
  const params = useParams();
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const isLocal = backend.kind === "local";

  const marketplacesQuery = useMarketplaces();
  const marketplaces = marketplacesQuery.data ?? [];
  const requestedName = params.marketplaceName ?? "";
  const selected =
    marketplaces.find((m) => m.name === requestedName) ??
    (requestedName
      ? null
      : (marketplaces.find((m) => m.is_default) ?? marketplaces[0] ?? null));

  const catalogQuery = useMarketplaceCatalog(selected?.name ?? "");
  const refreshMarketplace = useRefreshMarketplace();
  const removeMarketplace = useRemoveMarketplace();
  const updateMarketplace = useUpdateMarketplace();
  const manageEntry = useManageMarketplaceEntry();

  const [tab, setTab] = React.useState<KindTab>("all");
  const [filter, setFilter] = React.useState<MarketplaceFilterState>(
    EMPTY_MARKETPLACE_FILTER,
  );
  const [showAddModal, setShowAddModal] = React.useState(false);
  const [confirmRemove, setConfirmRemove] = React.useState(false);

  // Filters belong to one marketplace; switching resets them.
  const selectedName = selected?.name ?? "";
  React.useEffect(() => {
    setTab("all");
    setFilter(EMPTY_MARKETPLACE_FILTER);
  }, [selectedName]);

  const entries = catalogQuery.data?.entries ?? [];
  const counts = selected?.counts ?? EMPTY_COUNTS;
  const groups = React.useMemo(
    () => buildFacetGroups(entries, tab, filter),
    [entries, tab, filter],
  );
  const visible = React.useMemo(
    () => filterEntries(entries, tab, filter),
    [entries, tab, filter],
  );
  const activeFacets = countActiveFacets(filter);

  const busyEntryId = manageEntry.isPending
    ? manageEntry.variables?.entry.id
    : undefined;
  const marketplaceBusy =
    refreshMarketplace.isPending ||
    removeMarketplace.isPending ||
    updateMarketplace.isPending;

  const selectMarketplace = (name: string) => navigate(marketplacePath(name));

  const handleEntryAction = (
    entry: CatalogEntry,
    action: EntryAction,
    enabled?: boolean,
  ) => manageEntry.mutate({ action, entry, enabled });

  const unknownMarketplace =
    marketplacesQuery.isSuccess && requestedName.length > 0 && !selected;

  return (
    <div data-testid="marketplaces-screen" className="flex h-full lg:pl-10">
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-275 flex-col gap-6">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.MARKETPLACES$TITLE)}
              </h2>
              <p className="max-w-2xl text-sm text-tertiary-light">
                {t(I18nKey.MARKETPLACES$PAGE_DESCRIPTION)}
              </p>
            </div>
            <BrandButton
              type="button"
              variant="secondary"
              testId="marketplaces-add-button"
              isDisabled={!isLocal}
              className="flex-shrink-0 whitespace-nowrap"
              onClick={() => setShowAddModal(true)}
            >
              {t(I18nKey.MARKETPLACES$ADD_BUTTON)}
            </BrandButton>
          </div>

          {!isLocal && (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.MARKETPLACES$LOCAL_ONLY)}
              </p>
            </div>
          )}

          {isLocal && marketplacesQuery.isLoading && (
            <div className="h-24 rounded-2xl bg-tertiary animate-pulse" />
          )}

          {isLocal && marketplacesQuery.isError && (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {marketplacesQuery.error instanceof Error
                  ? marketplacesQuery.error.message
                  : t(I18nKey.ERROR$GENERIC)}
              </p>
              <BrandButton
                type="button"
                variant="secondary"
                className="mt-3"
                onClick={() => marketplacesQuery.refetch()}
              >
                {t(I18nKey.BUTTON$REFRESH)}
              </BrandButton>
            </div>
          )}

          {isLocal &&
            marketplacesQuery.isSuccess &&
            marketplaces.length === 0 && (
              <div
                data-testid="marketplaces-empty"
                className={extensionModuleEmptyStateClassName}
              >
                <p className="text-sm text-tertiary-light">
                  {t(I18nKey.MARKETPLACES$EMPTY)}
                </p>
              </div>
            )}

          {isLocal && marketplaces.length > 0 && (
            <MarketplacePickerBar
              marketplaces={marketplaces}
              selected={selected}
              isBusy={marketplaceBusy}
              onSelect={selectMarketplace}
              onToggleTrusted={() =>
                selected &&
                updateMarketplace.mutate({
                  name: selected.name,
                  request: { trusted: !selected.trusted },
                })
              }
              onRefresh={() =>
                selected && refreshMarketplace.mutate(selected.name)
              }
              onRemove={() => setConfirmRemove(true)}
            />
          )}

          {unknownMarketplace && (
            <div
              data-testid="marketplace-not-found"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.MARKETPLACES$NOT_FOUND, { name: requestedName })}
              </p>
            </div>
          )}

          {isLocal && selected && (
            <>
              <MarketplaceKindTabs
                active={tab}
                counts={counts}
                onChange={setTab}
              />

              {catalogQuery.isLoading && (
                <div className="flex flex-col gap-4">
                  {[1, 2, 3].map((i) => (
                    <div
                      key={i}
                      className="h-24 rounded-2xl bg-tertiary animate-pulse"
                    />
                  ))}
                </div>
              )}

              {catalogQuery.isError && (
                <div
                  data-testid="marketplace-catalog-error"
                  className={extensionModuleEmptyStateClassName}
                >
                  <p className="text-sm text-tertiary-light">
                    {catalogQuery.error instanceof Error
                      ? catalogQuery.error.message
                      : t(I18nKey.ERROR$GENERIC)}
                  </p>
                </div>
              )}

              {catalogQuery.data && (
                <>
                  <CatalogErrors errors={catalogQuery.data.errors} />

                  <MarketplaceCatalogToolbar
                    search={filter.query}
                    onSearchChange={(query) =>
                      setFilter((current) => ({ ...current, query }))
                    }
                  />

                  <div className="flex min-w-0 gap-6">
                    <MarketplaceFacetRail
                      groups={groups}
                      onToggle={(groupId, value) =>
                        setFilter((current) =>
                          toggleFacet(current, groupId, value),
                        )
                      }
                      className="hidden w-60 shrink-0 self-start md:flex"
                    />

                    <section className="flex min-w-0 flex-1 flex-col gap-3">
                      <div className="flex items-center justify-between gap-3 text-xs text-tertiary-light">
                        <span data-testid="marketplace-result-summary">
                          {t(I18nKey.MARKETPLACES$RESULT_COUNT, {
                            count: visible.length,
                          })}
                        </span>
                        {activeFacets > 0 ? (
                          <button
                            type="button"
                            data-testid="marketplace-clear-filters"
                            onClick={() =>
                              setFilter((current) => ({
                                ...EMPTY_MARKETPLACE_FILTER,
                                query: current.query,
                              }))
                            }
                            className="cursor-pointer underline hover:text-contrast"
                          >
                            {t(I18nKey.SETTINGS$SKILLS_CLEAR_FILTERS)}
                          </button>
                        ) : null}
                      </div>

                      {visible.length === 0 ? (
                        <div
                          data-testid="marketplace-no-match"
                          className={extensionModuleEmptyStateClassName}
                        >
                          <p className="text-sm text-tertiary-light">
                            {t(I18nKey.MARKETPLACES$NO_MATCH)}
                          </p>
                        </div>
                      ) : (
                        <div
                          className={extensionModuleCardGridContainerClassName}
                        >
                          <div className={extensionModuleCardGridClassName}>
                            {visible.map((entry) => (
                              <CatalogEntryCard
                                key={entry.id}
                                entry={entry}
                                marketplaceTrusted={selected.trusted}
                                isBusy={busyEntryId === entry.id}
                                onAction={(action, enabled) =>
                                  handleEntryAction(entry, action, enabled)
                                }
                              />
                            ))}
                          </div>
                        </div>
                      )}
                    </section>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </main>

      {showAddModal && (
        <AddMarketplaceModal
          onClose={() => setShowAddModal(false)}
          onAdded={(marketplace) => selectMarketplace(marketplace.name)}
        />
      )}

      {confirmRemove && selected && (
        <ConfirmationModal
          text={t(I18nKey.MARKETPLACES$REMOVE_CONFIRM, { name: selected.name })}
          confirmText={t(I18nKey.MARKETPLACES$REMOVE)}
          isConfirming={removeMarketplace.isPending}
          onCancel={() => setConfirmRemove(false)}
          onConfirm={() =>
            removeMarketplace.mutate(selected.name, {
              onSuccess: () => navigate(MARKETPLACES_PATH),
              onSettled: () => setConfirmRemove(false),
            })
          }
        />
      )}
    </div>
  );
}
