import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function OrgDefaultsBanner() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-tertiary-alt">
        {t(I18nKey.SETTINGS$ORG_DEFAULTS_INFO)}
      </p>
    </div>
  );
}
