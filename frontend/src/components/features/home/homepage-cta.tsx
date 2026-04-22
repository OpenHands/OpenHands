import { useTranslation } from "react-i18next";
import { Dispatch, SetStateAction } from "react";
import { Card } from "#/ui/card";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { setCTADismissed } from "#/utils/local-storage";
import { useTracking } from "#/hooks/use-tracking";
import CloseIcon from "#/icons/close.svg?react";

interface HomepageCTAProps {
  setShouldShowCTA: Dispatch<SetStateAction<boolean>>;
}

export function HomepageCTA({ setShouldShowCTA }: HomepageCTAProps) {
  const { t } = useTranslation();
  const { trackSaasSelfhostedInquiry } = useTracking();

  const handleClose = () => {
    setCTADismissed("homepage");
    setShouldShowCTA(false);
  };

  const handleLearnMoreClick = () => {
    trackSaasSelfhostedInquiry({ location: "home_page" });
  };

  return (
    <Card theme="dark" className={cn("w-[320px] cta-card-gradient")}>
      <button
        type="button"
        onClick={handleClose}
        className={cn(
          "absolute top-3 right-3 size-7 rounded-full",
          "border border-[#242424] bg-[#0A0A0A]",
          "flex items-center justify-center",
          "text-white/60 hover:text-white cursor-pointer",
          "shadow-[0px_1px_2px_-1px_#0000001A,0px_1px_3px_0px_#0000001A]",
        )}
        aria-label="Close"
      >
        <CloseIcon width={16} height={16} />
      </button>

      <div className="p-6 flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <CardTitle className="font-inter font-semibold text-xl leading-7 tracking-normal text-[#FAFAFA]">
            {t(I18nKey.CTA$ENTERPRISE_TITLE)}
          </CardTitle>

          <Typography.Text className="font-inter font-normal text-sm leading-5 tracking-normal text-[#8C8C8C]">
            {t(I18nKey.CTA$ENTERPRISE_DESCRIPTION)}
          </Typography.Text>
        </div>

        <a
          data-testid="homepage-cta-learn-more"
          href="https://openhands.dev/enterprise/"
          target="_blank"
          rel="noopener noreferrer"
          onClick={handleLearnMoreClick}
          className={cn(
            "inline-flex items-center justify-center",
            "w-fit h-10 px-4 rounded",
            "bg-[#050505] border border-[#242424]",
            "text-white hover:bg-white hover:text-black",
            "font-semibold text-sm",
          )}
        >
          {t(I18nKey.CTA$LEARN_MORE)}
        </a>
      </div>
    </Card>
  );
}
