import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { I18nKey } from "#/i18n/declaration";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import {
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";

interface CanvasExtensionCardProps {
  extension: InstalledCanvasExtensionInfo;
  isBusy: boolean;
  onToggle: () => void;
  onUninstall: () => void;
}

export function CanvasExtensionCard({
  extension,
  isBusy,
  onToggle,
  onUninstall,
}: CanvasExtensionCardProps) {
  const { t } = useTranslation("openhands");
  const pages = extension.manifest?.contributes?.pages ?? [];
  const displayName = extension.manifest?.display_name || extension.name;

  return (
    <article
      data-testid={`canvas-extension-card-${extension.name}`}
      className={`flex min-w-0 flex-col gap-4 p-4 ${extensionModuleCardSurfaceClassName}`}
    >
      <header className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-white">
            {displayName}
          </h3>
          <p className="mt-0.5 truncate text-xs text-tertiary-alt">
            {extension.source}
          </p>
        </div>
        <ToggleSwitch
          enabled={extension.enabled}
          label={t(
            extension.enabled ? I18nKey.COMMON$DISABLE : I18nKey.COMMON$ENABLE,
          )}
          onToggle={onToggle}
          className={isBusy ? "pointer-events-none opacity-50" : undefined}
        />
      </header>

      {extension.description || extension.manifest?.description ? (
        <p className="text-xs leading-relaxed text-tertiary-light">
          {extension.description || extension.manifest?.description}
        </p>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <span className={extensionModuleCardPillClassName}>
          {extension.enabled
            ? t(I18nKey.SETTINGS$SKILLS_ENABLED)
            : t(I18nKey.SETTINGS$SKILLS_DISABLED)}
        </span>
        <span className={extensionModuleCardPillClassName}>
          {t(I18nKey.SETTINGS$SKILLS_VERSION, {
            version: extension.version,
          })}
        </span>
        {pages.length ? (
          <span className={extensionModuleCardPillClassName}>
            {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_PAGES)}: {pages.length}
          </span>
        ) : null}
      </div>

      <dl className="grid min-w-0 grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        {extension.resolved_ref ? (
          <>
            <dt className="text-tertiary-alt">
              {t(I18nKey.SETTINGS$PLUGINS_REF_LABEL)}
            </dt>
            <dd className="min-w-0 truncate font-mono text-tertiary-light">
              {extension.resolved_ref}
            </dd>
          </>
        ) : null}
        {extension.repo_path ? (
          <>
            <dt className="text-tertiary-alt">
              {t(I18nKey.SETTINGS$PLUGINS_REPO_PATH_LABEL)}
            </dt>
            <dd className="min-w-0 truncate text-tertiary-light">
              {extension.repo_path}
            </dd>
          </>
        ) : null}
      </dl>

      {pages.length ? (
        <section className="flex min-w-0 flex-col gap-2">
          <ul className="flex flex-wrap gap-2">
            {pages.map((page) => (
              <li key={page.id} className={extensionModuleCardPillClassName}>
                {page.title}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <footer className="flex justify-end gap-2 border-t border-[var(--oh-border)] pt-3">
        <BrandButton
          type="button"
          variant="ghost-danger"
          testId={`canvas-extension-uninstall-${extension.name}`}
          isDisabled={isBusy}
          onClick={onUninstall}
        >
          {t(I18nKey.SETTINGS$PLUGINS_UNINSTALL)}
        </BrandButton>
      </footer>
    </article>
  );
}
