import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { ArrowLeft, ShieldAlert, ShieldCheck } from "lucide-react";
import {
  ExtensionsNavigation,
  ExtensionsCompactNavigation,
} from "#/components/features/skills/extensions-navigation";
import { CatalogEntryCard } from "#/components/features/marketplaces/catalog-entry-card";
import { CatalogErrors } from "#/components/features/marketplaces/catalog-errors";
import {
  MarketplaceCatalogToolbar,
  type InstallFilter,
  type KindFilter,
} from "#/components/features/marketplaces/marketplace-catalog-toolbar";
import { MARKETPLACE_SCOPE_LABEL_KEYS } from "#/components/features/marketplaces/catalog-kinds";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { ToggleSwitch } from "#/ui/toggle-switch";
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
  extensionModuleCardPillClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { CatalogEntry, CatalogKind } from "#/types/marketplaces";

const MARKETPLACES_PATH = "/marketplaces";

function matchesSearch(entry: CatalogEntry, query: string): boolean {
  if (!query) return true;
  const haystack = [entry.name, entry.description, ...entry.tags]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

export default function MarketplaceDetailScreen() {
  const { t } = useTranslation("openhands");
  const params = useParams();
  const marketplaceName = params.marketplaceName ?? "";
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const isLocal = backend.kind === "local";

  const query = useMarketplaceCatalog(marketplaceName);
  const refreshMarketplace = useRefreshMarketplace();
  const removeMarketplace = useRemoveMarketplace();
  const updateMarketplace = useUpdateMarketplace();
  const manageEntry = useManageMarketplaceEntry();

  const [search, setSearch] = React.useState("");
  const [kind, setKind] = React.useState<KindFilter>("all");
  const [installFilter, setInstallFilter] =
    React.useState<InstallFilter>("all");
  const [confirmRemove, setConfirmRemove] = React.useState(false);

  const catalog = query.data;
  const marketplace = catalog?.marketplace;
  const entries = catalog?.entries ?? [];

  const counts = React.useMemo(() => {
    const result: Record<CatalogKind, number> = {
      mcp: 0,
      skill: 0,
      plugin: 0,
      app: 0,
      automation: 0,
    };
    for (const entry of entries) result[entry.kind] += 1;
    return result;
  }, [entries]);

  const installedCount = entries.filter(
    (entry) => entry.install.state !== "not_installed",
  ).length;

  const visible = entries.filter(
    (entry) =>
      (kind === "all" || entry.kind === kind) &&
      (installFilter === "all" || entry.install.state !== "not_installed") &&
      matchesSearch(entry, search.trim()),
  );

  const busyEntryId = manageEntry.isPending
    ? manageEntry.variables?.entry.id
    : undefined;
  const marketplaceBusy =
    refreshMarketplace.isPending ||
    removeMarketplace.isPending ||
    updateMarketplace.isPending;

  const handleEntryAction = (
    entry: CatalogEntry,
    action: EntryAction,
    enabled?: boolean,
  ) => {
    manageEntry.mutate({ action, entry, enabled });
  };

  const notFound =
    query.isError &&
    typeof query.error === "object" &&
    query.error !== null &&
    "status" in query.error &&
    (query.error as { status?: number }).status === 404;

  return (
    <div
      data-testid="marketplace-detail-screen"
      className="flex h-full gap-4 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-200 flex-col gap-6">
          <ExtensionsCompactNavigation />

          <button
            type="button"
            data-testid="marketplace-back"
            onClick={() => navigate(MARKETPLACES_PATH)}
            className="flex w-fit cursor-pointer items-center gap-1 text-xs text-tertiary-alt hover:text-contrast"
          >
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
            {t(I18nKey.MARKETPLACES$BACK_TO_LIST)}
          </button>

          {!isLocal && (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.MARKETPLACES$LOCAL_ONLY)}
              </p>
            </div>
          )}

          {isLocal && query.isLoading && (
            <div className="flex flex-col gap-4">
              <div className="h-20 rounded-2xl bg-tertiary animate-pulse" />
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-24 rounded-2xl bg-tertiary animate-pulse"
                />
              ))}
            </div>
          )}

          {isLocal && query.isError && (
            <div
              data-testid="marketplace-detail-error"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {notFound
                  ? t(I18nKey.MARKETPLACES$NOT_FOUND, { name: marketplaceName })
                  : query.error instanceof Error
                    ? query.error.message
                    : t(I18nKey.ERROR$GENERIC)}
              </p>
            </div>
          )}

          {isLocal && marketplace && (
            <>
              <header className="flex min-w-0 flex-col gap-3">
                <div className="flex min-w-0 items-start justify-between gap-4">
                  <div className="min-w-0 space-y-1">
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <h2 className="truncate text-xl font-semibold leading-6 text-foreground">
                        {marketplace.name}
                      </h2>
                      <span className={extensionModuleCardPillClassName}>
                        {t(MARKETPLACE_SCOPE_LABEL_KEYS[marketplace.scope])}
                      </span>
                      {marketplace.is_default ? (
                        <span className={extensionModuleCardPillClassName}>
                          {t(I18nKey.MARKETPLACES$DEFAULT_BADGE)}
                        </span>
                      ) : null}
                    </div>
                    <p
                      className="truncate text-sm text-tertiary-light"
                      title={marketplace.source}
                    >
                      {marketplace.source}
                      {marketplace.ref ? ` @ ${marketplace.ref}` : ""}
                      {marketplace.resolved_ref
                        ? ` (${marketplace.resolved_ref})`
                        : ""}
                      {marketplace.last_refreshed_at
                        ? ` · ${t(I18nKey.MARKETPLACES$LAST_REFRESHED, {
                            when: new Date(
                              marketplace.last_refreshed_at,
                            ).toLocaleString(),
                          })}`
                        : ""}
                    </p>
                  </div>
                  <div className="flex shrink-0 gap-2">
                    <BrandButton
                      type="button"
                      variant="secondary"
                      testId="marketplace-detail-refresh"
                      isDisabled={marketplaceBusy}
                      onClick={() =>
                        refreshMarketplace.mutate(marketplace.name)
                      }
                    >
                      {t(I18nKey.BUTTON$REFRESH)}
                    </BrandButton>
                    {!marketplace.is_default ? (
                      <BrandButton
                        type="button"
                        variant="ghost-danger"
                        testId="marketplace-detail-remove"
                        isDisabled={marketplaceBusy}
                        onClick={() => setConfirmRemove(true)}
                      >
                        {t(I18nKey.BUTTON$DELETE)}
                      </BrandButton>
                    ) : null}
                  </div>
                </div>

                <div
                  data-testid="marketplace-trust-panel"
                  className="flex items-start justify-between gap-4 rounded-lg border border-border p-3"
                >
                  <div className="min-w-0">
                    <p className="flex items-center gap-1.5 text-sm text-contrast">
                      {marketplace.trusted ? (
                        <ShieldCheck
                          className="h-4 w-4 text-success"
                          aria-hidden
                        />
                      ) : (
                        <ShieldAlert
                          className="h-4 w-4 text-warning"
                          aria-hidden
                        />
                      )}
                      {t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}
                    </p>
                    <p className="mt-1 text-xs text-tertiary-light">
                      {t(I18nKey.MARKETPLACES$TRUSTED_HELP)}
                    </p>
                  </div>
                  <ToggleSwitch
                    enabled={marketplace.trusted}
                    label={t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}
                    disabled={marketplaceBusy}
                    onToggle={() =>
                      updateMarketplace.mutate({
                        name: marketplace.name,
                        request: { trusted: !marketplace.trusted },
                      })
                    }
                  />
                </div>
              </header>

              <CatalogErrors errors={catalog.errors} />

              <MarketplaceCatalogToolbar
                search={search}
                onSearchChange={setSearch}
                kind={kind}
                onKindChange={setKind}
                counts={counts}
                installFilter={installFilter}
                onInstallFilterChange={setInstallFilter}
                installedCount={installedCount}
              />

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
                <section
                  className={cn(
                    "flex min-w-0 flex-col gap-3",
                    extensionModuleCardGridContainerClassName,
                  )}
                >
                  <div className={extensionModuleCardGridClassName}>
                    {visible.map((entry) => (
                      <CatalogEntryCard
                        key={entry.id}
                        entry={entry}
                        marketplaceTrusted={marketplace.trusted}
                        isBusy={busyEntryId === entry.id}
                        onAction={(action, enabled) =>
                          handleEntryAction(entry, action, enabled)
                        }
                      />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </main>

      {confirmRemove && marketplace && (
        <ConfirmationModal
          text={t(I18nKey.MARKETPLACES$REMOVE_CONFIRM, {
            name: marketplace.name,
          })}
          confirmText={t(I18nKey.BUTTON$DELETE)}
          isConfirming={removeMarketplace.isPending}
          onCancel={() => setConfirmRemove(false)}
          onConfirm={() =>
            removeMarketplace.mutate(marketplace.name, {
              onSuccess: () => navigate(MARKETPLACES_PATH),
              onSettled: () => setConfirmRemove(false),
            })
          }
        />
      )}
    </div>
  );
}
