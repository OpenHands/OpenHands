import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface GitControlBarTooltipWrapperProps {
  tooltipMessage: string;
  testId: string;
  children: React.ReactNode;
  shouldShowTooltip: boolean;
}

export function GitControlBarTooltipWrapper({
  children,
  tooltipMessage,
  testId,
  shouldShowTooltip,
}: GitControlBarTooltipWrapperProps) {
  if (!shouldShowTooltip) {
    return children;
  }

  return (
    <StyledTooltip
      content={tooltipMessage}
      placement="top"
      showArrow
      tooltipClassName="bg-white text-black"
    >
      <span data-testid={testId} className="hover:opacity-100">
        {children}
      </span>
    </StyledTooltip>
  );
}
