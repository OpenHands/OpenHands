import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface ChatActionTooltipProps {
  children: React.ReactNode;
  tooltip: string | React.ReactNode;
  ariaLabel?: string;
}

export function ChatActionTooltip({
  children,
  tooltip,
  ariaLabel,
}: ChatActionTooltipProps) {
  return (
    <StyledTooltip
      content={tooltip}
      placement="bottom"
      tooltipClassName="bg-white text-black text-xs font-medium leading-5"
    >
      <span data-aria-label={ariaLabel}>{children}</span>
    </StyledTooltip>
  );
}
