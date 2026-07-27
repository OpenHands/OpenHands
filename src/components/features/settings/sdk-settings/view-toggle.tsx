import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsView } from "#/utils/sdk-settings-schema";
import { cn } from "#/utils/utils";
import { formControlTransitionClassName } from "#/utils/form-control-classes";

interface ViewToggleProps {
  view: SettingsView;
  setView: (view: SettingsView) => void;
  showAdvanced: boolean;
  showAll: boolean;
  isDisabled?: boolean;
}

const tabButtonClass = (isActive: boolean, isDisabled: boolean) =>
  cn(
    "w-fit px-2 py-2 text-sm cursor-pointer rounded-none bg-transparent",
    formControlTransitionClassName,
    "border-b-2 pb-2",
    isActive
      ? "text-white border-white"
      : "text-[var(--oh-muted)] border-transparent hover:text-white",
    isDisabled && "pointer-events-none opacity-30 cursor-not-allowed",
  );

export function ViewToggle({
  view,
  setView,
  showAdvanced,
  showAll,
  isDisabled = false,
}: ViewToggleProps) {
  const { t } = useTranslation("openhands");

  if (!showAdvanced && !showAll) return null;

  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className="mb-6 flex items-center gap-2"
    >
      <button
        data-testid="sdk-section-basic-toggle"
        type="button"
        role="tab"
        aria-selected={view === "basic"}
        disabled={isDisabled}
        className={tabButtonClass(view === "basic", isDisabled)}
        onClick={() => setView("basic")}
      >
        {t(I18nKey.SETTINGS$BASIC)}
      </button>
      {showAdvanced ? (
        <button
          data-testid="sdk-section-advanced-toggle"
          type="button"
          role="tab"
          aria-selected={view === "advanced"}
          disabled={isDisabled}
          className={tabButtonClass(view === "advanced", isDisabled)}
          onClick={() => setView("advanced")}
        >
          {t(I18nKey.SETTINGS$ADVANCED)}
        </button>
      ) : null}
      {showAll ? (
        <button
          data-testid="sdk-section-all-toggle"
          type="button"
          role="tab"
          aria-selected={view === "all"}
          disabled={isDisabled}
          className={tabButtonClass(view === "all", isDisabled)}
          onClick={() => setView("all")}
        >
          {t(I18nKey.SETTINGS$ALL)}
        </button>
      ) : null}
    </div>
  );
}
