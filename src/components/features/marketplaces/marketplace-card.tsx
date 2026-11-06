import React from "react";
import { useTranslation } from "react-i18next";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { CATALOG_KINDS, type MarketplaceSource } from "#/types/marketplaces";
import {
  CATALOG_KIND_LABEL_KEYS,
  MARKETPLACE_SCOPE_LABEL_KEYS,
} from "./catalog-kinds";

interface MarketplaceCardProps {
  marketplace: MarketplaceSource;
  isBusy?: boolean;
  onOpen: () => void;
  onRefresh: () => void;
  onRemove: () => void;
}

export function MarketplaceCard({
  marketplace,
  isBusy = false,
  onOpen,
  onRefresh,
  onRemove,
}: MarketplaceCardProps) {
  const { t } = useTranslation("openhands");

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen();
    }
  };

  const stop =
    (action: () => void) => (event?: React.MouseEvent<HTMLButtonElement>) => {
      event?.stopPropagation();
      action();
    };

  const total = CATALOG_KINDS.reduce(
    (sum, kind) => sum + marketplace.counts[kind],
    0,
  );

  return (
    <div
      data-testid={`marketplace-card-${marketplace.name}`}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={handleKeyDown}
      className={cn(
        "flex min-w-0 flex-col gap-3 overflow-hidden p-4",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-contrast">
              {marketplace.name}
            </h3>
            {marketplace.is_default ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.MARKETPLACES$DEFAULT_BADGE)}
              </span>
            ) : null}
          </div>
          <p
            className="mt-0.5 min-w-0 truncate text-xs text-tertiary-alt"
            title={marketplace.source}
          >
            {marketplace.source}
            {marketplace.ref ? ` @ ${marketplace.ref}` : ""}
            {marketplace.resolved_ref ? ` (${marketplace.resolved_ref})` : ""}
          </p>
        </div>
        <span
          data-testid={`marketplace-trust-${marketplace.name}`}
          className={cn(
            extensionModuleCardPillClassName,
            "gap-1",
            marketplace.trusted ? "text-success" : "text-warning",
          )}
        >
          {marketplace.trusted ? (
            <ShieldCheck className="h-3 w-3" aria-hidden />
          ) : (
            <ShieldAlert className="h-3 w-3" aria-hidden />
          )}
          {t(
            marketplace.trusted
              ? I18nKey.MARKETPLACES$TRUSTED_BADGE
              : I18nKey.MARKETPLACES$UNTRUSTED_BADGE,
          )}
        </span>
      </header>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className={extensionModuleCardPillClassName}>
          {t(MARKETPLACE_SCOPE_LABEL_KEYS[marketplace.scope])}
        </span>
        {CATALOG_KINDS.filter((kind) => marketplace.counts[kind] > 0).map(
          (kind) => (
            <span
              key={kind}
              data-testid={`marketplace-count-${marketplace.name}-${kind}`}
              className={extensionModuleCardPillClassName}
            >
              {marketplace.counts[kind]} {t(CATALOG_KIND_LABEL_KEYS[kind])}
            </span>
          ),
        )}
        {marketplace.error_count > 0 ? (
          <span
            className={cn(extensionModuleCardPillClassName, "text-warning")}
          >
            {t(I18nKey.MARKETPLACES$ERRORS_COUNT, {
              count: marketplace.error_count,
            })}
          </span>
        ) : null}
      </div>

      <footer className="flex items-center justify-between gap-3 text-xs text-tertiary-light">
        <span className="truncate">
          {t(I18nKey.MARKETPLACES$ENTRIES_COUNT, { count: total })}
          {marketplace.last_refreshed_at
            ? ` · ${t(I18nKey.MARKETPLACES$LAST_REFRESHED, {
                when: new Date(marketplace.last_refreshed_at).toLocaleString(),
              })}`
            : ""}
        </span>
        <div className="flex shrink-0 gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            testId={`marketplace-refresh-${marketplace.name}`}
            isDisabled={isBusy}
            onClick={stop(onRefresh)}
          >
            {t(I18nKey.BUTTON$REFRESH)}
          </BrandButton>
          {!marketplace.is_default ? (
            <BrandButton
              type="button"
              variant="ghost-danger"
              testId={`marketplace-remove-${marketplace.name}`}
              isDisabled={isBusy}
              onClick={stop(onRemove)}
            >
              {t(I18nKey.BUTTON$DELETE)}
            </BrandButton>
          ) : null}
        </div>
      </footer>
    </div>
  );
}
