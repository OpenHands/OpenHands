import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { Card } from "#/ui/card";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import StackedIcon from "#/icons/stacked.svg?react";
import { useTracking } from "#/hooks/use-tracking";

type LoginCTAProps = {
  className?: string;
  source?: "login_page" | "device_verify";
};

const ENTERPRISE_URL = "https://openhands.dev/enterprise";
const INFORMATION_REQUEST_PATH = "/information-request";

export function LoginCTA({
  className,
  source = "login_page",
}: LoginCTAProps = {}) {
  const { t } = useTranslation();
  const { trackSaasSelfhostedInquiry } = useTracking();
  const isDeviceVerifySource = source === "device_verify";
  const learnMoreButtonClassName = cn(
    "inline-flex items-center justify-center",
    "h-10 px-4 rounded",
    "bg-[#050505] border border-[#242424]",
    "text-white hover:bg-white hover:text-black",
    "font-semibold text-sm",
    "transition-colors",
  );

  const handleLearnMoreClick = () => {
    trackSaasSelfhostedInquiry({ location: source });
  };

  return (
    <Card
      testId="login-cta"
      theme="dark"
      className={cn(
        "w-full max-w-80 h-auto flex-col",
        "cta-card-gradient",
        className,
      )}
    >
      <div className={cn("flex h-full flex-col gap-[11px] p-6")}>
        <div className={cn("size-10")}>
          <StackedIcon width={40} height={40} />
        </div>

        <CardTitle>{t(I18nKey.CTA$ENTERPRISE)}</CardTitle>

        <Typography.Text className="text-[#8C8C8C] font-inter font-normal text-sm leading-5">
          {t(I18nKey.CTA$ENTERPRISE_DEPLOY)}
        </Typography.Text>

        <ul
          className={cn(
            "text-[#8C8C8C] font-inter font-normal text-sm leading-5 list-disc list-inside flex flex-col gap-1",
          )}
        >
          <li>{t(I18nKey.CTA$FEATURE_ON_PREMISES)}</li>
          <li>{t(I18nKey.CTA$FEATURE_DATA_CONTROL)}</li>
          <li>{t(I18nKey.CTA$FEATURE_COMPLIANCE)}</li>
          <li>{t(I18nKey.CTA$FEATURE_SUPPORT)}</li>
        </ul>

        <div className={cn("mt-auto h-10 flex justify-start")}>
          {/* Use <a> for external destination; react-router <Link> is only for internal app routes. */}
          {isDeviceVerifySource ? (
            <a
              href={ENTERPRISE_URL}
              target="_blank"
              rel="noopener noreferrer"
              onClick={handleLearnMoreClick}
              className={learnMoreButtonClassName}
            >
              {t(I18nKey.CTA$LEARN_MORE)}
            </a>
          ) : (
            <Link
              to={INFORMATION_REQUEST_PATH}
              onClick={handleLearnMoreClick}
              className={learnMoreButtonClassName}
            >
              {t(I18nKey.CTA$LEARN_MORE)}
            </Link>
          )}
        </div>
      </div>
    </Card>
  );
}
