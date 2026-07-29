import type { MarketplacePlugin } from "#/api/plugins-service";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";

interface PluginPickerCardProps {
  plugin: MarketplacePlugin;
  isSelected: boolean;
  isDisabled?: boolean;
  onToggle: (selected: boolean) => void;
}

/** A single selectable plugin in the picker catalog (display + attach toggle). */
export function PluginPickerCard({
  plugin,
  isSelected,
  isDisabled = false,
  onToggle,
}: PluginPickerCardProps) {
  return (
    <div
      data-testid={`plugin-picker-card-${plugin.name}`}
      className={cn(
        "flex min-w-0 items-start justify-between gap-3 rounded-lg bg-tertiary p-4",
      )}
    >
      <div className="min-w-0 flex-1">
        <h3
          data-testid={`plugin-picker-name-${plugin.name}`}
          className="truncate text-sm font-semibold text-white"
        >
          {plugin.name}
        </h3>
        <p
          className="mt-0.5 min-w-0 truncate text-xs text-tertiary-alt"
          title={plugin.source}
        >
          {plugin.source}
        </p>
        {plugin.description ? (
          <p className="mt-1 line-clamp-2 text-xs text-tertiary-light">
            {plugin.description}
          </p>
        ) : null}
        {plugin.repo_path || plugin.ref ? (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {plugin.repo_path ? (
              <span className={extensionModuleCardPillClassName}>
                {plugin.repo_path}
              </span>
            ) : null}
            {plugin.ref ? (
              <span className={extensionModuleCardPillClassName}>
                @{plugin.ref}
              </span>
            ) : null}
          </div>
        ) : null}
      </div>
      <CirclePlusCheckToggle
        testId={`plugin-picker-toggle-${plugin.name}`}
        isSelected={isSelected}
        isDisabled={isDisabled}
        onToggle={onToggle}
        enableLabelKey={I18nKey.BUTTON$ADD}
        disableLabelKey={I18nKey.COMMON$REMOVE}
        enableTooltipKey={I18nKey.BUTTON$ADD}
        disableTooltipKey={I18nKey.COMMON$REMOVE}
      />
    </div>
  );
}
