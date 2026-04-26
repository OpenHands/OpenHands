import { useTranslation } from "react-i18next";
import { Spinner } from "#/components/shared/spinner";
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
      <Spinner
        size="xl"
        label={t(I18nKey.HOME$LOADING)}
        className="text-white"
        labelClassName="text-2xl font-normal leading-5 p-4"
      />
    </div>
  );
}
