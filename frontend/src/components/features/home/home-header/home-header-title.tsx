import { useTranslation } from "react-i18next";

export function HomeHeaderTitle() {
  const { t } = useTranslation();

  return (
    <div className="h-[80px] flex flex-col items-center justify-center gap-2">
      <h1
        className="text-[34px] md:text-[44px] font-bold text-white tracking-[-0.03em] leading-tight text-center"
        style={{ fontFamily: "Inter, -apple-system, sans-serif" }}
      >
        {t("HOME$LETS_START_BUILDING")}
      </h1>
      <p className="text-[15px] text-[#71717A] font-normal tracking-[-0.01em] text-center">
        Describe what you want to build. neww.ai handles the rest.
      </p>
    </div>
  );
}
