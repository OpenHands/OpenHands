import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import InfoCircleIcon from "#/icons/info-circle.svg?react";

interface InfoTooltipProps {
  /** Explanatory text shown on hover. Also used as the icon's aria-label. */
  content: string;
}

/**
 * A small info icon that reveals an explanatory tooltip on hover. Centralizes
 * the tooltip's width, font size, and icon styling so settings tooltips stay
 * visually consistent (and long text wraps instead of stretching full-width).
 */
export function InfoTooltip({ content }: InfoTooltipProps) {
  return (
    <StyledTooltip
      content={content}
      placement="top"
      tooltipClassName="max-w-xs whitespace-normal text-xs"
    >
      <InfoCircleIcon
        width={14}
        height={14}
        className="text-tertiary-alt"
        aria-label={content}
      />
    </StyledTooltip>
  );
}
