import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const SPINNER_SIZES = {
  sm: "h-4 w-4 border-2",
  md: "h-5 w-5 border-2",
  lg: "h-12 w-12 border-4",
  xl: "h-16 w-16 border-4",
} as const;

interface SpinnerProps {
  size?: keyof typeof SPINNER_SIZES;
  label?: string;
  className?: string;
  spinnerClassName?: string;
  labelClassName?: string;
  testId?: string;
}

export function Spinner({
  size = "md",
  label,
  className,
  spinnerClassName,
  labelClassName,
  testId,
}: SpinnerProps) {
  const { t } = useTranslation();

  return (
    <div
      data-testid={testId}
      role="status"
      aria-live="polite"
      className={cn(
        "flex items-center justify-center gap-2 text-primary",
        label && "flex-col",
        className,
      )}
    >
      <div
        aria-hidden="true"
        className={cn(
          "animate-spin rounded-full border-solid border-current border-t-transparent",
          SPINNER_SIZES[size],
          spinnerClassName,
        )}
      />
      {label ? (
        <span className={labelClassName}>{label}</span>
      ) : (
        <span className="sr-only">{t(I18nKey.HOME$LOADING)}</span>
      )}
    </div>
  );
}
