import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { useHomeRecommendedAutomationsDismissed } from "#/components/features/home/use-home-recommended-automations-dismissed";
import { writeHomeRecommendedAutomationsDismissed } from "#/components/features/home/home-recommended-automations-storage";
import { I18nKey } from "#/i18n/declaration";

export function RecommendedAutomationsSwitch() {
  const { t } = useTranslation("openhands");
  const isDismissed = useHomeRecommendedAutomationsDismissed();

  const handleToggle = (showRail: boolean) => {
    writeHomeRecommendedAutomationsDismissed(!showRail);
  };

  return (
    <SettingsSwitch
      testId="show-recommended-automations-switch"
      isToggled={!isDismissed}
      onToggle={handleToggle}
    >
      {t(I18nKey.SETTINGS$SHOW_RECOMMENDED_AUTOMATIONS)}
    </SettingsSwitch>
  );
}
