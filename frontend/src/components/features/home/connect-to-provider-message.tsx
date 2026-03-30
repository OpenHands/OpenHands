import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useSettings } from "#/hooks/query/use-settings";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { I18nKey } from "#/i18n/declaration";

export function ConnectToProviderMessage() {
  const { isLoading } = useSettings();
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-4 justify-between h-full">
      <div className="flex flex-col gap-2.5">
        <div className="flex items-center gap-[10px]">
          <RepoForkedIcon width={24} height={24} />
          <span className="leading-5 font-bold text-base text-content">
            {t(I18nKey.COMMON$OPEN_REPOSITORY)}
          </span>
        </div>
        <p>{t("HOME$CONNECT_PROVIDER_MESSAGE")}</p>
      </div>
      <Link
        data-testid="navigate-to-settings-button"
        to="/settings/integrations"
        className="self-start w-full"
      >
        <BrandButton
          type="button"
          variant="primary"
          isDisabled={isLoading}
          className="w-full font-semibold"
        >
          {!isLoading && t("SETTINGS$TITLE")}
          {isLoading && t("HOME$LOADING")}
        </BrandButton>
      </Link>
    </div>
  );
}
