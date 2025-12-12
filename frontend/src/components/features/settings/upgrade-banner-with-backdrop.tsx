import { useTranslation } from "react-i18next";
import { UpgradeBanner } from "#/components/features/settings";

interface UpgradeBannerWithBackdropProps {
  onUpgradeClick: () => void;
  isDisabled?: boolean;
}

export function UpgradeBannerWithBackdrop({
  onUpgradeClick,
  isDisabled,
}: UpgradeBannerWithBackdropProps) {
  const { t } = useTranslation();

  return (
    <>
      <UpgradeBanner
        message={t("SETTINGS$UPGRADE_BANNER_MESSAGE")}
        onUpgradeClick={onUpgradeClick}
        className="sticky top-0 z-30 mb-6"
        isDisabled={isDisabled}
      />
      <div
        data-testid="settings-backdrop"
        className="absolute inset-0 z-20 opacity-50 bg-[#26282D]"
      />
    </>
  );
}
