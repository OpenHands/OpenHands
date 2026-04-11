import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Card } from "#/ui/card";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import StackedIcon from "#/icons/stacked.svg?react";
import { useTracking } from "#/hooks/use-tracking";

export function ContextMenuCTA() {
  const { t } = useTranslation();
  const { trackSaasSelfhostedInquiry } = useTracking();

  const handleLearnMoreClick = () => {
    trackSaasSelfhostedInquiry({ location: "context_menu" });
  };

  return (
    <Card
      testId="context-menu-cta"
      theme="dark"
      className={cn(
        "w-[286px] min-h-[200px] rounded-[6px]",
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
            onClick={handleLearnMoreClick}
            className={cn(
              "inline-flex items-center justify-center",
              "h-[40px] px-4 rounded-[4px]",
              "bg-[#050505] border border-[#242424]",
              "text-white hover:bg-white hover:text-black",
              "font-semibold text-sm",
            )}
          >
            {t(I18nKey.CTA$LEARN_MORE)}
          </a>
        </div>
      </div>
    </Card>
  );
}
