import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

type ConversationLoadingProps = {
  className?: string;
};

export function ConversationLoading({ className }: ConversationLoadingProps) {
  const { t } = useTranslation();

  return (
    <div
      className={cn(
        "bg-[#25272D] flex flex-col items-center justify-center h-full w-full",
        className,
      )}
    >
      <LoadingSpinner className="w-16 h-16 text-white" />
      <span className="text-2xl font-normal leading-5 text-white p-4">
        {t(I18nKey.HOME$LOADING)}
      </span>
    </div>
  );
}
