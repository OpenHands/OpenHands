import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface ProPillProps {
  className?: string;
}

export function ProPill({ className }: ProPillProps) {
  const { t } = useTranslation();

  return (
    <span
      className={cn(
        "bg-gold-dark border border-gold-text text-gold-text text-[8px] font-medium px-1.5 py-0.5 rounded-full whitespace-nowrap",
        className,
      )}
    >
      {t(I18nKey.SETTINGS$PRO_PILL)}
    </span>
  );
}
