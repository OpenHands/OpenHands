import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";

export function BackLink() {
  const { t } = useTranslation("openhands");

  return (
    <BackNavButton to="/automations">
      {t(I18nKey.AUTOMATIONS$DETAIL$BACK_TO_LIST)}
    </BackNavButton>
  );
}
