import React from "react";
import { ArrowRight, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { MetricsModal } from "#/components/features/conversation/metrics-modal/metrics-modal";
import { useContextWindowUsage } from "#/hooks/use-context-window-usage";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { I18nKey } from "#/i18n/declaration";
import { Divider } from "#/ui/divider";
import { cn } from "#/utils/utils";
import {
  formatCompactTokenCount,
  getContextWindowUsagePercentage,
} from "#/utils/format-token-count";
import { ContextWindowRing } from "./context-window-ring";

export function ContextWindowMeter() {
  const { t } = useTranslation("openhands");
  const usage = useContextWindowUsage();
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
  const [metricsModalOpen, setMetricsModalOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLDivElement>(
    () => setIsPopoverOpen(false),
    triggerRef,
  );

  if (!usage) {
    return null;
  }

  const usagePercentage = getContextWindowUsagePercentage(
    usage.perTurnToken,
    usage.contextWindow,
  );
  const roundedPercentage = Math.round(usagePercentage);
  const usageSummary = `${formatCompactTokenCount(usage.perTurnToken)} / ${formatCompactTokenCount(usage.contextWindow)} (${roundedPercentage}%)`;

  const openMetricsModal = () => {
    setIsPopoverOpen(false);
    setMetricsModalOpen(true);
  };

  return (
    <>
      <div className="relative shrink-0">
        <button
          ref={triggerRef}
          type="button"
          className="flex size-8 items-center justify-center rounded-full hover:bg-white/10 transition-colors"
          aria-label={t(I18nKey.CHAT_INTERFACE$CONTEXT_WINDOW_METER_LABEL)}
          aria-expanded={isPopoverOpen}
          aria-haspopup="dialog"
          data-testid="context-window-meter"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            setIsPopoverOpen((open) => !open);
          }}
        >
          <ContextWindowRing percentage={usagePercentage} />
        </button>

        {isPopoverOpen && (
          <div
            ref={popoverRef}
            data-testid="context-window-meter-popover"
            className={cn(
              "absolute bottom-full right-0 z-[60] mb-2 w-[280px]",
              "rounded-md border border-[var(--oh-border-subtle)] bg-tertiary p-3 shadow-lg",
            )}
          >
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="text-[var(--oh-muted)]">
                  {t(I18nKey.CONVERSATION$CONTEXT_WINDOW)}
                </span>
                <span className="inline-flex items-center gap-1 text-[var(--oh-foreground)]">
                  {usageSummary}
                  <ChevronRight
                    width={14}
                    height={14}
                    className="text-[var(--oh-muted)]"
                    aria-hidden
                  />
                </span>
              </div>

              <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--oh-border)]">
                <div
                  className="h-full rounded-full bg-white transition-all duration-300"
                  style={{ width: `${Math.min(100, usagePercentage)}%` }}
                />
              </div>
            </div>

            <Divider inset="menu" className="my-2" />

            <button
              type="button"
              className="flex w-full items-center justify-between gap-2 rounded px-1 py-1 text-sm text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] transition-colors"
              data-testid="context-window-plan-usage"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                openMetricsModal();
              }}
            >
              <span>{t(I18nKey.CHAT_INTERFACE$PLAN_USAGE)}</span>
              <ArrowRight width={14} height={14} aria-hidden />
            </button>
          </div>
        )}
      </div>

      <MetricsModal
        isOpen={metricsModalOpen}
        onOpenChange={setMetricsModalOpen}
      />
    </>
  );
}
