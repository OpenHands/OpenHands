import { LoaderCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

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
      <LoaderCircle className="animate-spin w-16 h-16" color="white" />
      <span className="text-2xl font-normal leading-5 text-white p-4">
        {t(I18nKey.HOME$LOADING)}
      </span>
    </div>
  );
}
