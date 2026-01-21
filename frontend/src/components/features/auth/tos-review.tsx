import { Dispatch, SetStateAction, useState } from "react";
import { useTranslation } from "react-i18next";

interface LoginOrgSelectorProps {
  setIsOrgSelected: Dispatch<SetStateAction<boolean>>;
  setIsTOSReviewComplete: Dispatch<SetStateAction<boolean>>;
}

function TOSReview({
  setIsOrgSelected,
  setIsTOSReviewComplete,
}: LoginOrgSelectorProps) {
  const { t } = useTranslation();
  const buttonLabelClasses = "text-sm font-medium leading-5 px-1";

  const [hasAcceptedTOS, setHasAcceptedTOS] = useState(false);

  // TODO:: fetch TOS text

  const handleSubmit = () => {
    if (!hasAcceptedTOS) return;

    // TODO:
    // - persist TOS acceptance in DB
    // - trigger login
    // - show loader + redirect user to home screen

    setIsTOSReviewComplete(true);
  };

  return (
    <div className="flex flex-col gap-[32px]">
      <span className="text-[39px] font-medium text-center">
        {t("Review Terms Of Service")}
      </span>

      <span className="text-center text-[14px] font-[400]">
        {t("Please review and accept our Terms of Service before continuing.")}
      </span>

      <div className="flex flex-col gap-[8px] w-full">
        <div className="h-[373px] rounded-[5px] bg-[#454545] border border-[#727987] pt-[10px] pb-[10px] pr-[10px] pl-[15px]" />

        <label className="flex items-center gap-[8px] cursor-pointer">
          <input
            type="checkbox"
            checked={hasAcceptedTOS}
            onChange={(e) => setHasAcceptedTOS(e.target.checked)}
            className="h-[16px] w-[16px] rounded-[4px] border-[#D4D4D4] bg-[#FFFFFF] accent-white"
          />
          <span className="text-[12px] font-[400] text-[#A3A3A3]">
            {t("I accept the Terms of Service")}
          </span>
        </label>
      </div>

      <div className="flex gap-[6px] h-[40px] text-[14px] w-full">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!hasAcceptedTOS}
          className="flex-1 bg-[#FFFFFF] rounded-[4px] text-[#000000] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span className={buttonLabelClasses}>{t("Next")}</span>
        </button>

        <button
          type="button"
          onClick={() => setIsOrgSelected(false)}
          className="flex-1 border border-[#FFFFFF] rounded-[4px]"
        >
          <span className={buttonLabelClasses}>{t("Back")}</span>
        </button>
      </div>
    </div>
  );
}

export default TOSReview;
