import { useTranslation } from "react-i18next";
import { ArrowUpCircle, TerminalSquare } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import {
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import type { CatalogEntry } from "#/types/marketplaces";
import type { EntryAction } from "#/hooks/mutation/use-manage-marketplace-entries";
import { CATALOG_KIND_LABEL_KEYS } from "./catalog-kinds";

interface CatalogEntryCardProps {
  entry: CatalogEntry;
  /** The marketplace this entry comes from is marked trusted. */
  marketplaceTrusted: boolean;
  isBusy?: boolean;
  onAction: (action: EntryAction, enabled?: boolean) => void;
}

/** What "install" means for each kind, in the button label. */
const INSTALL_LABEL_KEY: Record<CatalogEntry["kind"], I18nKey> = {
  mcp: I18nKey.MARKETPLACES$ACTION_ADD_MCP,
  skill: I18nKey.MARKETPLACES$ACTION_INSTALL,
  plugin: I18nKey.MARKETPLACES$ACTION_INSTALL,
  app: I18nKey.MARKETPLACES$ACTION_INSTALL,
  automation: I18nKey.MARKETPLACES$ACTION_USE_TEMPLATE,
};

const INSTALLED_LABEL_KEY: Record<CatalogEntry["kind"], I18nKey> = {
  mcp: I18nKey.MARKETPLACES$ACTION_MCP_ADDED,
  skill: I18nKey.MARKETPLACES$ACTION_INSTALLED,
  plugin: I18nKey.MARKETPLACES$ACTION_INSTALLED,
  app: I18nKey.MARKETPLACES$ACTION_INSTALLED,
  automation: I18nKey.MARKETPLACES$ACTION_TEMPLATE_USED,
};

export function CatalogEntryCard({
  entry,
  marketplaceTrusted,
  isBusy = false,
  onAction,
}: CatalogEntryCardProps) {
  const { t } = useTranslation("openhands");
  const installed = entry.install.state !== "not_installed";
  const updateAvailable = entry.install.state === "update_available";
  const trustBlocked = entry.runs_code && !marketplaceTrusted && !installed;
  /** Enabling makes sense for things the agent loads; templates become automations. */
  const canToggle = installed && entry.kind !== "automation";

  return (
    <div
      data-testid={`catalog-entry-${entry.kind}-${entry.name}`}
      className={cn(
        "flex min-w-0 flex-col gap-3 overflow-hidden p-4",
        extensionModuleCardSurfaceClassName,
      )}
    >
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">
            <span
              data-testid={`catalog-entry-kind-${entry.kind}-${entry.name}`}
              className={cn(extensionModuleCardPillClassName, "text-contrast")}
            >
              {t(CATALOG_KIND_LABEL_KEYS[entry.kind])}
            </span>
            <h3 className="min-w-0 truncate text-sm font-semibold text-contrast">
              {entry.name}
            </h3>
            {entry.version ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.SETTINGS$SKILLS_VERSION, { version: entry.version })}
              </span>
            ) : null}
          </div>
        </div>

        {canToggle ? (
          <CirclePlusCheckToggle
            testId={`catalog-entry-toggle-${entry.kind}-${entry.name}`}
            isSelected={entry.install.enabled}
            isDisabled={isBusy}
            onToggle={(enabled) => onAction("enable", enabled)}
            enableLabelKey={I18nKey.MARKETPLACES$ENABLE}
            disableLabelKey={I18nKey.MARKETPLACES$DISABLE}
            disableTooltipKey={I18nKey.MARKETPLACES$DISABLE}
          />
        ) : null}
      </header>

      {entry.description ? (
        <p className="line-clamp-2 break-words text-xs leading-relaxed text-tertiary-light">
          {entry.description}
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-1.5">
        {entry.runs_code ? (
          <span
            data-testid={`catalog-entry-runs-code-${entry.kind}-${entry.name}`}
            className={cn(
              extensionModuleCardPillClassName,
              "gap-1 text-warning",
            )}
            title={t(I18nKey.MARKETPLACES$RUNS_CODE_HELP)}
          >
            <TerminalSquare className="h-3 w-3" aria-hidden />
            {t(I18nKey.MARKETPLACES$RUNS_CODE)}
          </span>
        ) : null}
        {entry.transport ? (
          <span className={extensionModuleCardPillClassName}>
            {entry.transport}
          </span>
        ) : null}
        {entry.requires.length > 0 ? (
          <span className={extensionModuleCardPillClassName}>
            {t(I18nKey.MARKETPLACES$REQUIRES, {
              items: entry.requires.join(", "),
            })}
          </span>
        ) : null}
        {entry.also_in.map((other) => (
          <span
            key={other}
            data-testid={`catalog-entry-also-in-${entry.kind}-${entry.name}`}
            className={extensionModuleCardPillClassName}
          >
            {t(I18nKey.MARKETPLACES$ALSO_IN, { marketplace: other })}
          </span>
        ))}
        {entry.tags.slice(0, 4).map((tag) => (
          <span key={tag} className={extensionModuleCardPillClassName}>
            {tag}
          </span>
        ))}
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-2">
        <span className="min-w-0 truncate text-xs text-tertiary-light">
          {installed && entry.install.installed_ref
            ? t(I18nKey.MARKETPLACES$INSTALLED_FROM, {
                ref: entry.install.installed_ref,
              })
            : trustBlocked
              ? t(I18nKey.MARKETPLACES$TRUST_REQUIRED)
              : ""}
        </span>
        <div className="flex shrink-0 gap-2">
          {!installed ? (
            <BrandButton
              type="button"
              variant="secondary"
              testId={`catalog-entry-install-${entry.kind}-${entry.name}`}
              isDisabled={isBusy || trustBlocked}
              onClick={() => onAction("install")}
            >
              {t(INSTALL_LABEL_KEY[entry.kind])}
            </BrandButton>
          ) : (
            <>
              <span
                data-testid={`catalog-entry-installed-${entry.kind}-${entry.name}`}
                className={cn(extensionModuleCardPillClassName, "text-success")}
              >
                {t(INSTALLED_LABEL_KEY[entry.kind])}
              </span>
              {updateAvailable ? (
                <BrandButton
                  type="button"
                  variant="primary"
                  testId={`catalog-entry-update-${entry.kind}-${entry.name}`}
                  isDisabled={isBusy}
                  startContent={
                    <ArrowUpCircle className="h-3.5 w-3.5" aria-hidden />
                  }
                  onClick={() => onAction("update")}
                >
                  {t(I18nKey.MARKETPLACES$ACTION_UPDATE)}
                </BrandButton>
              ) : null}
              <BrandButton
                type="button"
                variant="ghost-danger"
                testId={`catalog-entry-uninstall-${entry.kind}-${entry.name}`}
                isDisabled={isBusy}
                onClick={() => onAction("uninstall")}
              >
                {t(
                  entry.kind === "automation"
                    ? I18nKey.MARKETPLACES$ACTION_FORGET_TEMPLATE
                    : I18nKey.MARKETPLACES$ACTION_UNINSTALL,
                )}
              </BrandButton>
            </>
          )}
        </div>
      </footer>
    </div>
  );
}
