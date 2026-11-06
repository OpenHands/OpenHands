import { useTranslation } from "react-i18next";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import type { MarketplaceSource } from "#/types/marketplaces";
import { MARKETPLACE_SCOPE_LABEL_KEYS } from "./catalog-kinds";

interface MarketplacePickerBarProps {
  marketplaces: MarketplaceSource[];
  selected: MarketplaceSource | null;
  isBusy?: boolean;
  onSelect: (name: string) => void;
  onToggleTrusted: () => void;
  onRefresh: () => void;
  onRemove: () => void;
}

/**
 * The band above the catalog: choose a marketplace, see where it comes from,
 * and manage it (trust, refresh, delete). Adding one lives in the page header.
 */
export function MarketplacePickerBar({
  marketplaces,
  selected,
  isBusy = false,
  onSelect,
  onToggleTrusted,
  onRefresh,
  onRemove,
}: MarketplacePickerBarProps) {
  const { t } = useTranslation("openhands");

  return (
    <section
      data-testid="marketplace-picker-bar"
      className="flex flex-col gap-3 rounded-xl border border-border bg-base-secondary p-4"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <SettingsDropdownInput
          // The dropdown reads its items once; remount it when the set changes.
          key={marketplaces.map((m) => m.name).join("|")}
          testId="marketplace-picker"
          name="marketplace"
          label={t(I18nKey.MARKETPLACES$PICKER_LABEL)}
          wrapperClassName="md:max-w-sm"
          items={marketplaces.map((m) => ({
            key: m.name,
            label: `${m.name} · ${t(MARKETPLACE_SCOPE_LABEL_KEYS[m.scope])}`,
          }))}
          defaultSelectedKey={selected?.name}
          selectedKey={selected?.name}
          onSelectionChange={(key) => {
            if (key) onSelect(String(key));
          }}
        />
        {selected ? (
          <div className="flex flex-wrap items-center gap-2 md:ml-auto">
            <label
              data-testid="marketplace-trust-control"
              className="flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm text-contrast"
            >
              {selected.trusted ? (
                <ShieldCheck className="h-4 w-4 text-success" aria-hidden />
              ) : (
                <ShieldAlert className="h-4 w-4 text-warning" aria-hidden />
              )}
              <span>{t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}</span>
              <ToggleSwitch
                enabled={selected.trusted}
                label={t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}
                disabled={isBusy}
                onToggle={onToggleTrusted}
              />
            </label>
            <BrandButton
              type="button"
              variant="secondary"
              testId="marketplace-refresh"
              isDisabled={isBusy}
              onClick={onRefresh}
            >
              {t(I18nKey.BUTTON$REFRESH)}
            </BrandButton>
            {!selected.is_default ? (
              <BrandButton
                type="button"
                variant="secondary"
                testId="marketplace-remove"
                isDisabled={isBusy}
                onClick={onRemove}
              >
                {t(I18nKey.MARKETPLACES$REMOVE)}
              </BrandButton>
            ) : null}
          </div>
        ) : null}
      </div>

      {selected ? (
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-tertiary-light">
          <span className="min-w-0 truncate" title={selected.source}>
            {selected.source}
            {selected.ref ? ` @ ${selected.ref}` : ""}
            {selected.resolved_ref ? ` (${selected.resolved_ref})` : ""}
          </span>
          {selected.last_refreshed_at ? (
            <span className="shrink-0">
              ·{" "}
              {t(I18nKey.MARKETPLACES$LAST_REFRESHED, {
                when: new Date(selected.last_refreshed_at).toLocaleString(),
              })}
            </span>
          ) : null}
          {selected.is_default ? (
            <span className={extensionModuleCardPillClassName}>
              {t(I18nKey.MARKETPLACES$DEFAULT_BADGE)}
            </span>
          ) : null}
          {selected.token_secret ? (
            <span className={extensionModuleCardPillClassName}>
              {t(I18nKey.MARKETPLACES$TOKEN_BADGE, {
                secret: selected.token_secret,
              })}
            </span>
          ) : null}
          {selected.error_count > 0 ? (
            <span
              className={cn(extensionModuleCardPillClassName, "text-warning")}
            >
              {t(I18nKey.MARKETPLACES$ERRORS_COUNT, {
                count: selected.error_count,
              })}
            </span>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
