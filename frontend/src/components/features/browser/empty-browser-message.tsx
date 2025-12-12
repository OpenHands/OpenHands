import { useTranslation } from "react-i18next";
import { IoIosGlobe } from "react-icons/io";
import { I18nKey } from "#/i18n/declaration";

export function EmptyBrowserMessage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-10 gap-4">
      <IoIosGlobe size={113} />
      <span className="text-[#8D95A9] text-[19px] font-normal leading-5">
        {" "}
        {t(I18nKey.BROWSER$NO_PAGE_LOADED)}
      </span>
    </div>
  );
}
