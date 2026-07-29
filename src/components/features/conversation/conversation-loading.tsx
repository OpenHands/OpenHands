import { LoaderCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { TextShimmer } from "#/components/shared/text-shimmer";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

type ConversationLoadingProps = {
  className?: string;
};

export function ConversationLoading({ className }: ConversationLoadingProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      className={cn(
        "bg-[var(--oh-surface)] flex h-full w-full flex-col items-center justify-center gap-3",
        className,
      )}
    >
      <LoaderCircle
        className="h-8 w-8 shrink-0 animate-spin text-tertiary-light"
        aria-hidden
      />
      <TextShimmer
        as="p"
        role="status"
        aria-live="polite"
        className="block w-full text-center text-base font-normal leading-5"
        duration={1}
        spread={2}
      >
        {t(I18nKey.HOME$LOADING)}
      </TextShimmer>
    </div>
  );
}
