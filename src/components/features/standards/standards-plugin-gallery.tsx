import { useTranslation } from "react-i18next";
import { Shield } from "lucide-react";
import {
  STANDARDS_SOURCE_BUILTIN,
  STANDARDS_SOURCE_PROJECT,
  STANDARDS_SOURCE_USER,
} from "#/api/standards-service/standards-constants";
import type { StandardsPluginInfo } from "#/api/standards-service/standards-types";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { I18nKey } from "#/i18n/declaration";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";

const SOURCE_LABEL: Record<string, I18nKey> = {
  [STANDARDS_SOURCE_BUILTIN]: I18nKey.STANDARDS$SOURCE_BUILTIN,
  [STANDARDS_SOURCE_USER]: I18nKey.STANDARDS$SOURCE_USER,
  [STANDARDS_SOURCE_PROJECT]: I18nKey.STANDARDS$SOURCE_PROJECT,
};

export interface StandardsPluginGalleryProps {
  plugins: StandardsPluginInfo[];
  onToggle: (plugin: StandardsPluginInfo) => void;
}

export function StandardsPluginGallery({
  plugins,
  onToggle,
}: StandardsPluginGalleryProps) {
  const { t } = useTranslation("openhands");

  if (plugins.length === 0) {
    return (
      <p
        data-testid="standards-gallery-empty"
        className="text-sm text-tertiary-light"
      >
        {t(I18nKey.STANDARDS$NO_PLUGINS)}
      </p>
    );
  }

  return (
    <section
      data-testid="standards-plugin-gallery"
      className={extensionModuleCardGridContainerClassName}
    >
      <div className={extensionModuleCardGridClassName}>
        {plugins.map((plugin) => (
          <article
            key={plugin.name}
            data-testid={`standards-plugin-card-${plugin.name}`}
            className={`flex min-w-0 flex-col gap-3 p-4 ${extensionModuleCardSurfaceClassName}`}
          >
            <header className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                <Shield
                  className="mt-0.5 shrink-0 text-tertiary-light"
                  size={16}
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <h3 className="truncate text-sm font-semibold text-white">
                    {plugin.display_name}
                  </h3>
                  <p className="mt-1 text-xs leading-relaxed text-tertiary-light">
                    {plugin.description}
                  </p>
                </div>
              </div>
              <ToggleSwitch
                enabled={plugin.enabled}
                label={t(
                  plugin.enabled
                    ? I18nKey.COMMON$DISABLE
                    : I18nKey.COMMON$ENABLE,
                )}
                onToggle={() => onToggle(plugin)}
              />
            </header>
            <div className="flex flex-wrap gap-2">
              <span className={extensionModuleCardPillClassName}>
                {t(
                  SOURCE_LABEL[plugin.source] ??
                    I18nKey.STANDARDS$SOURCE_BUILTIN,
                )}
              </span>
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.STANDARDS$VERSION, { version: plugin.version })}
              </span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
