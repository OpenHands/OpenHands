import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import StackedIcon from "#/icons/stacked.svg?react";

export function ContextMenuCTA() {
  const { t } = useTranslation();

  return (
    <div
      className={cn(
        "w-[286px] h-[449px] rounded-[6px]",
        "border border-[#24242499]",
        "flex flex-col",
      )}
      style={{
        background:
          "linear-gradient(0deg, rgba(10, 10, 10, 0.5), rgba(10, 10, 10, 0.5)), radial-gradient(80% 60% at 50% 0%, rgba(255, 255, 255, 0.25) 0%, rgba(0, 0, 0, 0) 100%)",
        boxShadow: "0px 4px 6px -4px #0000001A, 0px 10px 15px -3px #0000001A",
      }}
    >
      <div
        className={cn(
          "w-[236px] h-[246px]",
          "flex flex-col gap-[11px]",
          "mt-[175px] ml-[25px]",
        )}
      >
        <div>
          <StackedIcon width={40} height={40} />
        </div>

        <div className="w-[236px]">
          <CardTitle>{t(I18nKey.CTA$ENTERPRISE_TITLE)}</CardTitle>
        </div>

        <div className="w-[236px]">
          <Typography.Text
            className={cn(
              "text-[#8C8C8C] font-inter font-normal",
              "text-[14px] leading-[20px]",
            )}
          >
            {t(I18nKey.CTA$ENTERPRISE_DESCRIPTION)}
          </Typography.Text>
        </div>

        <div className="w-[236px] h-[40px] flex justify-start mt-auto">
          <a
            href="https://openhands.dev/enterprise/"
            target="_blank"
            rel="noopener noreferrer"
          >
            <BrandButton
              variant="primary"
              type="button"
              className={cn(
                "w-[111px] h-[40px] rounded-[4px]",
                "bg-[#050505] border border-[#242424]",
                "text-white hover:bg-[#0a0a0a]",
              )}
            >
              {t(I18nKey.CTA$LEARN_MORE)}
            </BrandButton>
          </a>
        </div>
      </div>
    </div>
  );
}
