import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { useSidebarOnboardingChecklistDismissed } from "#/components/features/sidebar/use-sidebar-onboarding-checklist-dismissed";
import { writeSidebarOnboardingChecklistDismissed } from "#/components/features/sidebar/sidebar-onboarding-checklist-storage";
import { I18nKey } from "#/i18n/declaration";

export function GettingStartedChecklistSwitch() {
  const { t } = useTranslation("openhands");
  const isDismissed = useSidebarOnboardingChecklistDismissed();

  const handleToggle = (showChecklist: boolean) => {
    writeSidebarOnboardingChecklistDismissed(!showChecklist);
  };

  return (
    <SettingsSwitch
      testId="show-getting-started-checklist-switch"
      isToggled={!isDismissed}
      onToggle={handleToggle}
    >
      {t(I18nKey.SETTINGS$SHOW_GETTING_STARTED_CHECKLIST)}
    </SettingsSwitch>
  );
}
