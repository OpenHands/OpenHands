import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardPillClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import type { PluginViewModel } from "./build-plugins-view-model";

interface PluginCardProps {
  plugin: PluginViewModel;
  /** A mutation targeting this plugin is in flight. */
  isBusy?: boolean;
  /** Management actions are unavailable (e.g. non-local backend). */
  isDisabled?: boolean;
  onOpen: () => void;
  onInstall: () => void;
  onToggle: (enabled: boolean) => void;
}

export function PluginCard({
  plugin,
  isBusy = false,
  isDisabled = false,
  onOpen,
  onInstall,
  onToggle,
}: PluginCardProps) {
  const { t } = useTranslation("openhands");

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen();
    }
  };

  const handleInstall = (event?: React.MouseEvent<HTMLButtonElement>) => {
    event?.stopPropagation();
    onInstall();
  };

  return (
    <div
      data-testid={`plugin-card-${plugin.name}`}
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
          <h3
            data-testid={`plugin-name-${plugin.name}`}
            className="truncate text-sm font-semibold text-white"
          >
            {plugin.name}
          </h3>
          {plugin.source ? (
            <p
              data-testid={`plugin-source-${plugin.name}`}
              className="mt-0.5 min-w-0 truncate text-xs text-tertiary-alt"
              title={plugin.source}
            >
              {plugin.source}
            </p>
          ) : null}
        </div>

        {plugin.installed ? (
          <CirclePlusCheckToggle
            testId={`plugin-toggle-${plugin.name}`}
            isSelected={plugin.enabled}
            isDisabled={isDisabled || isBusy}
            onToggle={onToggle}
            disableTooltipKey={I18nKey.COMMON$DISABLE}
          />
        ) : plugin.isLocal ? (
          <span
            data-testid={`plugin-local-badge-${plugin.name}`}
            className={cn(extensionModuleCardPillClassName, "flex-shrink-0")}
          >
            {t(I18nKey.SETTINGS$PLUGINS_FILTER_LOCAL)}
          </span>
        ) : (
          <BrandButton
            type="button"
            variant="secondary"
            testId={`plugin-install-${plugin.name}`}
            isDisabled={isDisabled || isBusy}
            className="flex-shrink-0 whitespace-nowrap"
            onClick={handleInstall}
          >
            {t(
              isBusy
                ? I18nKey.SETTINGS$PLUGINS_INSTALLING
                : I18nKey.SETTINGS$PLUGINS_INSTALL,
            )}
          </BrandButton>
        )}
      </header>

      {plugin.description ? (
        <p
          data-testid={`plugin-description-${plugin.name}`}
          className="line-clamp-2 break-words text-xs leading-relaxed text-tertiary-light"
        >
          {plugin.description}
        </p>
      ) : null}

      {(plugin.installed || plugin.isLocal) && plugin.version ? (
        <span
          data-testid={`plugin-version-${plugin.name}`}
          className={cn(extensionModuleCardPillClassName, "self-start")}
        >
          {t(I18nKey.SETTINGS$SKILLS_VERSION, { version: plugin.version })}
        </span>
      ) : null}
    </div>
  );
}
