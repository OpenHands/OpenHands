import React from "react";
import { Gauge, Loader2, Minimize } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useContextWindowUsage } from "#/hooks/use-context-window-usage";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useCompactContextAction } from "#/hooks/use-compact-context-action";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { ConversationNameContextMenuIconText } from "#/components/features/conversation/conversation-name-context-menu-icon-text";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { Divider } from "#/ui/divider";
import { cn } from "#/utils/utils";
import {
  formatCompactTokenCount,
  getContextWindowUsagePercentage,
} from "#/utils/format-token-count";
import { getContextFillTone } from "#/components/features/conversation/usage-panel/context-meter";
import { ContextWindowRing } from "./context-window-ring";

const TONE_BAR_CLASS = {
  neutral: "bg-white",
  warning: "bg-amber-500",
  danger: "bg-red-500",
} as const;

const TONE_LABEL_CLASS = {
  neutral: "text-[var(--oh-muted)]",
  warning: "text-amber-500",
  danger: "text-red-500",
} as const;

export function ContextWindowMeter() {
  const { t } = useTranslation("openhands");
  const usage = useContextWindowUsage();
  const { navigateToTab } = useSelectConversationTab();
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLDivElement>(
    () => setIsPopoverOpen(false),
    triggerRef,
  );
  const { handleCompact, isCompacting, isDisabled } = useCompactContextAction(
    usage?.perTurnToken ?? 0,
  );

  if (!usage) {
    return null;
  }

  const usagePercentage = getContextWindowUsagePercentage(
    usage.perTurnToken,
    usage.contextWindow,
  );
  const roundedPercentage = Math.round(usagePercentage);
  const remainingPercentage = Math.max(0, 100 - roundedPercentage);
  const usagePercentLabel = `${roundedPercentage}% ${t(I18nKey.CONVERSATION$USED)} (${remainingPercentage}% ${t(I18nKey.CONVERSATION$LEFT)})`;
  const usageTokenSummary = `${formatCompactTokenCount(usage.perTurnToken)} / ${formatCompactTokenCount(usage.contextWindow)}`;
  const tone = getContextFillTone(usagePercentage);

  const openUsagePanel = () => {
    setIsPopoverOpen(false);
    navigateToTab("usage");
  };

  return (
    <div className="relative shrink-0">
      <StyledTooltip
        content={t(I18nKey.CHAT_INTERFACE$SHOW_CONTEXT)}
        placement="top"
      >
        <button
          ref={triggerRef}
          type="button"
          className="flex size-8 items-center justify-center rounded-full hover:bg-white/10 transition-colors"
          aria-label={`${t(I18nKey.CHAT_INTERFACE$CONTEXT_WINDOW_METER_LABEL)}: ${usagePercentLabel}`}
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
      </StyledTooltip>

      {isPopoverOpen && (
        <div
          ref={popoverRef}
          data-testid="context-window-meter-popover"
          className={cn(
            "absolute bottom-full right-0 z-[60] mb-2 w-[280px]",
            "flex flex-col gap-0.5 rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 shadow-lg",
          )}
        >
          <div className="flex flex-col gap-2 px-2 py-1.5">
            <div className="flex items-center justify-between gap-2 text-sm">
              <span className="font-semibold text-[var(--oh-foreground)]">
                {t(I18nKey.CONVERSATION$CONTEXT_WINDOW)}
              </span>
              <span className={cn("shrink-0 text-xs", TONE_LABEL_CLASS[tone])}>
                {usagePercentLabel}
              </span>
            </div>

            <button
              type="button"
              data-testid="context-window-meter-bar-button"
              className="relative h-1.5 w-full rounded-full bg-[var(--oh-border)] cursor-pointer"
              aria-label={t(I18nKey.COMMON$USAGE)}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                openUsagePanel();
              }}
            >
              <span
                className={cn(
                  "absolute inset-y-0 left-0 rounded-full transition-all duration-300",
                  TONE_BAR_CLASS[tone],
                )}
                style={{ width: `${Math.min(100, usagePercentage)}%` }}
              />
            </button>

            <div className="flex items-center justify-between gap-2">
              <button
                type="button"
                data-testid="context-window-compact-button"
                disabled={isDisabled}
                aria-busy={isCompacting}
                aria-label={t(I18nKey.CONVERSATION$COMPACT_CONTEXT)}
                className={cn(
                  "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs",
                  "text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]",
                  "transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                )}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  handleCompact();
                }}
              >
                {isCompacting ? (
                  <Loader2 className="size-3 animate-spin" aria-hidden />
                ) : (
                  <Minimize className="size-3" aria-hidden />
                )}
                <span>{t(I18nKey.CONVERSATION$COMPACT_CONTEXT)}</span>
              </button>
              <span className="text-xs text-[var(--oh-muted)]">
                {usageTokenSummary}
              </span>
            </div>
          </div>

          <Divider inset="menu" />

          <ContextMenuListItem
            testId="context-window-plan-usage"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              openUsagePanel();
            }}
          >
            <ConversationNameContextMenuIconText
              icon={<Gauge size={16} />}
              text={t(I18nKey.COMMON$USAGE)}
            />
          </ContextMenuListItem>
        </div>
      )}
    </div>
  );
}
