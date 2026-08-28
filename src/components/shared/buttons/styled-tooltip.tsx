import { Tooltip } from "@heroui/react";
import React, { ReactNode } from "react";
import { cn } from "#/utils/utils";

export type StyledTooltipPlacement =
  | "top"
  | "top start"
  | "top end"
  | "bottom"
  | "bottom start"
  | "bottom end"
  | "left"
  | "left top"
  | "left bottom"
  | "right"
  | "right top"
  | "right bottom"
  | "start"
  | "end";

export interface StyledTooltipProps {
  children: ReactNode;
  content: string | ReactNode;
  tooltipClassName?: React.HTMLAttributes<HTMLDivElement>["className"];
  placement?: StyledTooltipPlacement;
  showArrow?: boolean;
  closeDelay?: number;
  offset?: number;
  shouldFlip?: boolean;
}

function getTooltipTriggerChild(children: ReactNode) {
  if (React.Children.count(children) === 1 && React.isValidElement(children)) {
    return children;
  }
  return <span className="inline-flex">{children}</span>;
}

export function StyledTooltip({
  children,
  content,
  tooltipClassName,
  placement = "right",
  showArrow = false,
  closeDelay = 100,
  shouldFlip,
  offset = 7,
}: StyledTooltipProps) {
  const shouldSkipAnimation = import.meta.env.MODE === "test";

  return (
    <Tooltip
      delay={0}
      closeDelay={closeDelay}
      shouldSkipAnimation={shouldSkipAnimation}
    >
      <Tooltip.Trigger>{getTooltipTriggerChild(children)}</Tooltip.Trigger>
      <Tooltip.Content
        placement={placement}
        offset={offset}
        shouldFlip={shouldFlip}
        showArrow={showArrow}
        className={cn(
          "z-[9999] rounded-md px-2 py-1 text-xs font-medium shadow-md",
          "bg-white text-black",
          tooltipClassName,
        )}
      >
        {content}
      </Tooltip.Content>
    </Tooltip>
  );
}
