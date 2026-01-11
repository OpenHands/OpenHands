import { useTranslation } from "react-i18next";
import { FaCodeCompare } from "react-icons/fa6";
import { I18nKey } from "#/i18n/declaration";

export function EmptyChangesMessage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-10 gap-4">
      <FaCodeCompare size={100} className="text-muted" />
      <span className="text-subtle text-[19px] font-normal leading-5">
        {t(I18nKey.DIFF_VIEWER$NO_CHANGES)}
      </span>
    </div>
  );
}
