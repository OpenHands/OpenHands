import { Code2, Clock3 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { formControlTransitionClassName } from "#/utils/form-control-classes";

export type HomeLauncherMode = "code" | "automate";

const HOME_LAUNCHER_MODES: HomeLauncherMode[] = ["code", "automate"];

interface HomeLauncherModeToggleProps {
  mode: HomeLauncherMode;
  onChange: (mode: HomeLauncherMode) => void;
}

export function HomeLauncherModeToggle({
  mode,
  onChange,
}: HomeLauncherModeToggleProps) {
  const { t } = useTranslation("openhands");

  const getLabel = (item: HomeLauncherMode) =>
    item === "code"
      ? t(I18nKey.COMMON$CODE)
      : t(I18nKey.AUTOMATE$SECTION_TITLE);

  return (
    <div
      role="group"
      aria-label={t(I18nKey.HOME$MODE_TOGGLE_LABEL)}
      data-testid="home-launcher-mode-toggle"
      className="inline-flex items-center gap-1 rounded-full bg-white/10 p-1"
    >
      {HOME_LAUNCHER_MODES.map((item) => {
        const isSelected = item === mode;
        const Icon = item === "code" ? Code2 : Clock3;

        return (
          <button
            key={item}
            type="button"
            data-testid={`home-launcher-mode-${item}`}
            aria-pressed={isSelected}
            onClick={() => onChange(item)}
            className={cn(
              "inline-flex h-8 items-center gap-2 rounded-full px-4 text-sm leading-5",
              formControlTransitionClassName,
              isSelected
                ? "bg-white/10 text-white shadow-[0_1px_3px_rgba(0,0,0,0.24)]"
                : "text-tertiary-light hover:bg-white/5 hover:text-white",
            )}
          >
            <Icon aria-hidden className="h-4 w-4" strokeWidth={2} />
            <span>{getLabel(item)}</span>
          </button>
        );
      })}
    </div>
  );
}
