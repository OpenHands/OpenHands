import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { OrgWideSettingsBadge } from "#/components/features/settings/org-wide-settings-badge";

export function OrgDefaultsBanner() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-3">
      <OrgWideSettingsBadge />
      <p className="text-sm text-tertiary-alt">
        {t(I18nKey.SETTINGS$ORG_DEFAULTS_INFO)}
      </p>
    </div>
  );
}
