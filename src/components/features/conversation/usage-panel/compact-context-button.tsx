import { Info, Loader2, Minimize } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useCompactContextAction } from "#/hooks/use-compact-context-action";
import { CONTEXT_FILL_WARNING_PERCENT } from "./context-meter";

interface CompactContextButtonProps {
  /** Current context fill percentage; past the warning threshold the button becomes a prominent CTA. */
  fillPercent: number;
  /** Optional live per-turn token count used as a fallback snapshot before compaction. */
  perTurnToken?: number;
}

/**
 * "Compact context" action: POSTs `/api/conversations/{id}/condense` with a
 * spinner while in flight and toasts for start, completion (with tokens
 * freed), and failure. Disabled while the agent is actively running (the
 * server would race the active step) and promoted to a primary
 * call-to-action once the context fill passes
 * {@link CONTEXT_FILL_WARNING_PERCENT}.
 */
export function CompactContextButton({
  fillPercent,
  perTurnToken = 0,
}: CompactContextButtonProps) {
  const { t } = useTranslation("openhands");
  const { handleCompact, isCompacting, isDisabled, description } =
    useCompactContextAction(perTurnToken);
  const isHighFill = fillPercent > CONTEXT_FILL_WARNING_PERCENT;

  return (
    <div
      data-testid="compact-context-section"
      className="flex flex-col gap-2 border-t border-[var(--oh-border-subtle)] pt-3"
    >
      {isHighFill && (
        <span className="text-xs text-amber-500">
          {t(I18nKey.CONVERSATION$CONTEXT_FILLING_UP)}
        </span>
      )}
      <div className="flex items-center gap-2">
        <BrandButton
          testId="compact-context-button"
          type="button"
          variant={isHighFill ? "primary" : "secondary"}
          isDisabled={isDisabled}
          aria-busy={isCompacting}
          onClick={handleCompact}
          startContent={
            isCompacting ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <Minimize className="h-4 w-4" aria-hidden />
            )
          }
          className="w-fit"
        >
          {t(I18nKey.CONVERSATION$COMPACT_CONTEXT)}
        </BrandButton>
        <StyledTooltip
          content={description}
          placement="top"
          tooltipClassName="max-w-[220px] text-xs font-normal leading-relaxed whitespace-normal"
        >
          <button
            type="button"
            data-testid="compact-context-info"
            aria-label={description}
            className="flex size-5 shrink-0 items-center justify-center rounded-full text-[var(--oh-muted)] hover:text-[var(--oh-foreground)] transition-colors cursor-help"
          >
            <Info className="size-3.5" aria-hidden />
          </button>
        </StyledTooltip>
      </div>
    </div>
  );
}
