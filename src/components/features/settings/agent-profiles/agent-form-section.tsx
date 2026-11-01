import type { ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface AgentFormSectionProps {
  title: ReactNode;
  children: ReactNode;
  testId: string;
  optional?: boolean;
  open?: boolean;
  onToggle?: (open: boolean) => void;
}

export function AgentFormSection({
  title,
  children,
  testId,
  optional = true,
  open,
  onToggle,
}: AgentFormSectionProps) {
  const { t } = useTranslation("openhands");
  return (
    <details
      data-testid={testId}
      open={open}
      onToggle={(event) => onToggle?.(event.currentTarget.open)}
      className="group rounded-xl border border-border bg-surface-raised p-5"
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 text-base font-medium text-foreground [&::-webkit-details-marker]:hidden">
        <ChevronRight
          aria-hidden="true"
          className="size-4 shrink-0 transition-transform group-open:rotate-90"
        />
        <span>{title}</span>
        {optional && (
          <span className="text-sm font-normal text-tertiary-light">
            · {t(I18nKey.COMMON$OPTIONAL)}
          </span>
        )}
      </summary>
      <div className="flex flex-col gap-4 pt-5">{children}</div>
    </details>
  );
}
