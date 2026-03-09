import { useTranslation } from "react-i18next";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import StackedIcon from "#/icons/stacked.svg?react";

export function ContextMenuCTA() {
  const { t } = useTranslation();

  return (
    <div
      className="w-[286px] h-[449px] rounded-[6px] border border-[#24242499] flex flex-col"
      style={{
        background: `linear-gradient(0deg, rgba(10, 10, 10, 0.5), rgba(10, 10, 10, 0.5)), radial-gradient(237.19% 96.24% at 53.77% -1.6%, rgba(255, 255, 255, 0.14) 0%, rgba(0, 0, 0, 0) 55%)`,
      }}
    >
      <div className="w-[236px] h-[246px] flex flex-col gap-[11px] mt-[175px] ml-[25px]">
        <div>
          <StackedIcon width={40} height={40} />
        </div>

        <div className="w-[236px]">
          <CardTitle>{t(I18nKey.CTA$ENTERPRISE_TITLE)}</CardTitle>
        </div>

        <div className="w-[236px]">
          <Typography.Text className="text-[#8C8C8C] font-inter font-normal text-[14px] leading-[20px]">
            {t(I18nKey.CTA$ENTERPRISE_DESCRIPTION)}
          </Typography.Text>
        </div>

        <div className="w-[236px] h-[40px] flex justify-start mt-auto">
          <BrandButton
            variant="primary"
            type="button"
            className="w-[111px] h-[40px] rounded-[4px] bg-[#050505] border border-[#242424] text-white hover:bg-[#0a0a0a]"
          >
            {t(I18nKey.CTA$LEARN_MORE)}
          </BrandButton>
        </div>
      </div>
    </div>
  );
}
