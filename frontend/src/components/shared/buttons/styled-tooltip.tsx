import { Tooltip, TooltipProps } from "@heroui/react";
import React, { ReactNode } from "react";

export interface StyledTooltipProps {
  children: ReactNode;
  content: string | ReactNode;
  tooltipClassName?: React.HTMLAttributes<HTMLDivElement>["className"];
  placement?: TooltipProps["placement"];
  showArrow?: boolean;
  closeDelay?: number;
}

export function StyledTooltip({
  children,
  content,
  tooltipClassName,
  placement = "right",
  showArrow = false,
  closeDelay = 100,
}: StyledTooltipProps) {
  return (
    <Tooltip
      content={content}
      closeDelay={closeDelay}
      placement={placement}
      className={tooltipClassName}
      showArrow={showArrow}
    >
      <div className="inline-flex">{children}</div>
    </Tooltip>
  );
}
