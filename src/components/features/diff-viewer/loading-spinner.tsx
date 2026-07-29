import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

export interface LoadingSpinnerProps {
  className?: string;
}

export function LoadingSpinner({ className }: LoadingSpinnerProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex items-center justify-center">
      <div
        className={cn(
          "animate-spin rounded-full border-4 border-transparent border-t-white",
          className,
        )}
        role="status"
        aria-label={t(I18nKey.HOME$LOADING)}
      />
    </div>
  );
}
