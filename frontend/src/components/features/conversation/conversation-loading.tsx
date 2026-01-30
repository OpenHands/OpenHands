import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Spinner } from "#/ui/spinner";

export function ConversationLoading() {
  const { t } = useTranslation();

  return (
    <div className="bg-[#25272D] border border-[#525252] rounded-xl flex flex-col items-center justify-center h-full w-full">
      <Spinner
        size="xl"
        className="text-white"
        label={t(I18nKey.HOME$LOADING)}
        labelClassName="text-2xl font-normal leading-5 p-4"
      />
    </div>
  );
}
