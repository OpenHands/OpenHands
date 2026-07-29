import { useTranslation } from "react-i18next";
import PuzzleIcon from "#/icons/u-puzzle-piece.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  formControlBorderClassName,
  formControlSurfaceClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";

interface PluginPickerTriggerProps {
  /** Number of currently-attached plugins, shown as a badge when > 0. */
  count: number;
  onClick: () => void;
  disabled?: boolean;
}

/** Pill button that opens the plugin picker; mirrors `OpenLauncherButton`. */
export function PluginPickerTrigger({
  count,
  onClick,
  disabled = false,
}: PluginPickerTriggerProps) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid="open-plugin-picker"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex flex-row items-center gap-2 rounded-full px-2.5 py-1 text-white",
        formControlBorderClassName,
        formControlSurfaceClassName,
        formControlTransitionClassName,
        disabled
          ? "cursor-not-allowed opacity-50"
          : "cursor-pointer hover:bg-surface-raised",
      )}
    >
      <span className="flex h-4 w-4 shrink-0 items-center justify-center">
        <PuzzleIcon width={16} height={16} className="shrink-0" aria-hidden />
      </span>
      <span className="text-sm font-normal leading-5">
        {t(I18nKey.PLUGINS$PICKER_TRIGGER)}
      </span>
      {count > 0 ? (
        <span
          data-testid="plugin-picker-count"
          className="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-white px-1 text-[11px] font-semibold leading-none text-black"
        >
          {count}
        </span>
      ) : null}
    </button>
  );
}
