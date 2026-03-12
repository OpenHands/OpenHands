import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Card } from "#/ui/card";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import StackedIcon from "#/icons/stacked.svg?react";

export function ContextMenuCTA() {
  const { t } = useTranslation();

  return (
    <Card
      testId="context-menu-cta"
      theme="dark"
      className={cn(
        "w-[286px] min-h-[200px] h-full rounded-[6px]",
        "flex-col justify-end",
        "cta-card-gradient",
      )}
    >
      <div
        data-testid="context-menu-cta-content"
        className={cn("flex flex-col gap-[11px] p-[25px]")}
      >
        <StackedIcon width={40} height={40} />

        <CardTitle>{t(I18nKey.CTA$ENTERPRISE_TITLE)}</CardTitle>

        <Typography.Text
          className={cn(
            "text-[#8C8C8C] font-inter font-normal",
            "text-[14px] leading-[20px]",
          )}
        >
          {t(I18nKey.CTA$ENTERPRISE_DESCRIPTION)}
        </Typography.Text>

        <div className="flex mt-auto">
          <a
            href="https://openhands.dev/enterprise/"
            target="_blank"
            rel="noopener noreferrer"
          >
            <BrandButton
              variant="primary"
              type="button"
              className={cn(
                "h-[40px] px-4 rounded-[4px]",
                "bg-[#050505] border border-[#242424]",
                "text-white hover:bg-[#0a0a0a]",
              )}
            >
              {t(I18nKey.CTA$LEARN_MORE)}
            </BrandButton>
          </a>
        </div>
      </div>
    </Card>
  );
}
