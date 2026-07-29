import { Check, FilePlus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface PastedImageUploadAsFileButtonProps {
  active: boolean;
  onToggle: () => void;
}

export function PastedImageUploadAsFileButton({
  active,
  onToggle,
}: PastedImageUploadAsFileButtonProps) {
  const { t } = useTranslation("openhands");
  const uploadLabel = t(I18nKey.CHAT_INTERFACE$UPLOAD_IMAGES_AS_FILES);
  const doNotUploadLabel = t(I18nKey.CHAT_INTERFACE$DO_NOT_UPLOAD_AS_FILE);
  const label = active ? doNotUploadLabel : uploadLabel;

  return (
    <div className="absolute bottom-2 left-1 z-10 h-4 w-4">
      <StyledTooltip
        content={label}
        placement="bottom"
        offset={10}
        shouldFlip={false}
        tooltipClassName="bg-white text-black text-xs font-medium leading-5"
      >
        <button
          type="button"
          aria-label={label}
          aria-pressed={active}
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          className={cn(
            "flex h-4 w-4 items-center justify-center rounded-full bg-[var(--oh-surface)] text-[var(--oh-foreground)] transition-colors cursor-pointer hover:bg-[var(--oh-muted)]",
          )}
        >
          {active ? (
            <Check width={10} height={10} strokeWidth={2.5} aria-hidden />
          ) : (
            <FilePlus width={10} height={10} strokeWidth={2.5} aria-hidden />
          )}
        </button>
      </StyledTooltip>
    </div>
  );
}
