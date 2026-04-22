import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import InfoCircleIcon from "#/icons/info-circle.svg?react";

export function OrgWideSettingsBadge() {
  const { t } = useTranslation();

  return (
    <div
      data-testid="org-wide-settings-badge"
      className="flex items-center gap-2 bg-[rgba(31,31,31,0.4)] border border-[#242424] rounded-full px-2.5 py-1"
    >
      <InfoCircleIcon width={12} height={12} className="text-[#8c8c8c]" />
      <Typography.Text className="text-[11px] font-medium text-[#8c8c8c] leading-5">
        {t(I18nKey.SETTINGS$ORG_WIDE_SETTING_BADGE)}
      </Typography.Text>
    </div>
  );
}
