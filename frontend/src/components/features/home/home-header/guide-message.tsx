import { useTranslation } from "react-i18next";

export function GuideMessage() {
  const { t } = useTranslation();

  return (
    <div className="w-fit flex flex-col md:flex-row items-start md:items-center justify-center gap-1.5 rounded-full bg-[#18181B] border border-[#27272A] leading-5 text-[#A1A1AA] text-[13px] font-medium m-1 md:h-9 px-4 pb-0.5 md:px-4 md:py-0 shadow-sm">
      <span>{t("HOME$GUIDE_MESSAGE_TITLE")} </span>
      <a
        href="https://docs.all-hands.dev/usage/getting-started"
        target="_blank"
        rel="noopener noreferrer"
        className="text-[#818CF8] hover:text-[#A5B4FC] transition-colors"
      >
        <span className="underline underline-offset-2">{t("COMMON$CLICK_HERE")}</span>
      </a>
    </div>
  );
}
