import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { SettingsView } from "#/utils/sdk-settings-schema";

interface ViewToggleProps {
  view: SettingsView;
  setView: (view: SettingsView) => void;
  showAdvanced: boolean;
  showAll: boolean;
  isDisabled?: boolean;
  // Extra buttons rendered in the same flex row as Basic/Advanced/All,
  // placed after them. Used to slot section-level nav (e.g. a Profiles
  // button) into the same control strip as the view toggles.
  trailing?: React.ReactNode;
}

export function ViewToggle({
  view,
  setView,
  showAdvanced,
  showAll,
  isDisabled = false,
  trailing,
}: ViewToggleProps) {
  const { t } = useTranslation();

  const hasViewButtons = showAdvanced || showAll;
  if (!hasViewButtons && !trailing) return null;

  return (
    <div className="flex items-center gap-2 mb-6 flex-wrap">
      {hasViewButtons ? (
        <BrandButton
          testId="sdk-section-basic-toggle"
          variant={view === "basic" ? "primary" : "secondary"}
          type="button"
          isDisabled={isDisabled}
          onClick={() => setView("basic")}
        >
          {t(I18nKey.SETTINGS$BASIC)}
        </BrandButton>
      ) : null}
      {showAdvanced ? (
        <BrandButton
          testId="sdk-section-advanced-toggle"
          variant={view === "advanced" ? "primary" : "secondary"}
          type="button"
          isDisabled={isDisabled}
          onClick={() => setView("advanced")}
        >
          {t(I18nKey.SETTINGS$ADVANCED)}
        </BrandButton>
      ) : null}
      {showAll ? (
        <BrandButton
          testId="sdk-section-all-toggle"
          variant={view === "all" ? "primary" : "secondary"}
          type="button"
          isDisabled={isDisabled}
          onClick={() => setView("all")}
        >
          {t(I18nKey.SETTINGS$ALL)}
        </BrandButton>
      ) : null}
      {trailing}
    </div>
  );
}
