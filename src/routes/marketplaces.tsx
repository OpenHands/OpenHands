import React from "react";
import { useTranslation } from "react-i18next";
import {
  ExtensionsNavigation,
  ExtensionsCompactNavigation,
} from "#/components/features/skills/extensions-navigation";
import { MarketplaceCard } from "#/components/features/marketplaces/marketplace-card";
import { AddMarketplaceModal } from "#/components/features/marketplaces/add-marketplace-modal";
import {
  MARKETPLACE_SCOPE_LABEL_KEYS,
  marketplacePath,
} from "#/components/features/marketplaces/catalog-kinds";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useMarketplaces } from "#/hooks/query/use-marketplaces";
import {
  useRefreshMarketplace,
  useRemoveMarketplace,
} from "#/hooks/mutation/use-manage-marketplaces";
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
import type { MarketplaceScope, MarketplaceSource } from "#/types/marketplaces";

const SCOPE_ORDER: readonly MarketplaceScope[] = ["instance", "org", "user"];

export default function MarketplacesScreen() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const isLocal = backend.kind === "local";

  const query = useMarketplaces();
  const refreshMarketplace = useRefreshMarketplace();
  const removeMarketplace = useRemoveMarketplace();

  const [showAddModal, setShowAddModal] = React.useState(false);
  const [removing, setRemoving] = React.useState<MarketplaceSource | null>(
    null,
  );

  const marketplaces = query.data ?? [];
  const busyName =
    (refreshMarketplace.isPending && refreshMarketplace.variables) ||
    (removeMarketplace.isPending && removeMarketplace.variables) ||
    null;

  const grouped = SCOPE_ORDER.map((scope) => ({
    scope,
    items: marketplaces.filter((m) => m.scope === scope),
  })).filter((group) => group.items.length > 0);

  return (
    <div
      data-testid="marketplaces-screen"
      className="flex h-full gap-4 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-200 flex-col gap-6">
          <ExtensionsCompactNavigation />
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

          {isLocal && query.isLoading && (
            <div className="flex flex-col gap-4">
              {[1, 2].map((i) => (
                <div
                  key={i}
                  className="h-28 rounded-2xl bg-tertiary animate-pulse"
                />
              ))}
            </div>
          )}

          {isLocal && query.isError && (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {query.error instanceof Error
                  ? query.error.message
                  : t(I18nKey.ERROR$GENERIC)}
              </p>
              <BrandButton
                type="button"
                variant="secondary"
                className="mt-3"
                onClick={() => query.refetch()}
              >
                {t(I18nKey.BUTTON$REFRESH)}
              </BrandButton>
            </div>
          )}

          {isLocal && query.isSuccess && marketplaces.length === 0 && (
            <div
              data-testid="marketplaces-empty"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.MARKETPLACES$EMPTY)}
              </p>
            </div>
          )}

          {isLocal &&
            grouped.map((group) => (
              <section
                key={group.scope}
                data-testid={`marketplaces-scope-${group.scope}`}
                className={cn(
                  "flex min-w-0 flex-col gap-3",
                  extensionModuleCardGridContainerClassName,
                )}
              >
                <h3 className="text-xs font-medium uppercase tracking-wide text-tertiary-alt">
                  {t(MARKETPLACE_SCOPE_LABEL_KEYS[group.scope])}
                </h3>
                <div className={extensionModuleCardGridClassName}>
                  {group.items.map((marketplace) => (
                    <MarketplaceCard
                      key={marketplace.name}
                      marketplace={marketplace}
                      isBusy={busyName === marketplace.name}
                      onOpen={() => navigate(marketplacePath(marketplace.name))}
                      onRefresh={() =>
                        refreshMarketplace.mutate(marketplace.name)
                      }
                      onRemove={() => setRemoving(marketplace)}
                    />
                  ))}
                </div>
              </section>
            ))}
        </div>
      </main>

      {showAddModal && (
        <AddMarketplaceModal
          onClose={() => setShowAddModal(false)}
          onAdded={(marketplace) => navigate(marketplacePath(marketplace.name))}
        />
      )}

      {removing && (
        <ConfirmationModal
          text={t(I18nKey.MARKETPLACES$REMOVE_CONFIRM, {
            name: removing.name,
          })}
          confirmText={t(I18nKey.BUTTON$DELETE)}
          isConfirming={removeMarketplace.isPending}
          onCancel={() => setRemoving(null)}
          onConfirm={() =>
            removeMarketplace.mutate(removing.name, {
              onSettled: () => setRemoving(null),
            })
          }
        />
      )}
    </div>
  );
}
