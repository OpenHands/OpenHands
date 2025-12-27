import React, { ReactNode } from "react";
import { NavLink } from "react-router";
import { TooltipProps } from "@heroui/react";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "./styled-tooltip";

export interface TooltipButtonProps {
  children: ReactNode;
  tooltip: string | ReactNode;
  onClick?: () => void;
  href?: string;
  navLinkTo?: string;
  ariaLabel: string;
  testId?: string;
  className?: React.HTMLAttributes<HTMLButtonElement>["className"];
  tooltipClassName?: React.HTMLAttributes<HTMLDivElement>["className"];
  disabled?: boolean;
  placement?: TooltipProps["placement"];
  showArrow?: boolean;
}

export function TooltipButton({
  children,
  tooltip,
  onClick,
  href,
  navLinkTo,
  ariaLabel,
  testId,
  className,
  tooltipClassName,
  disabled = false,
  placement = "right",
  showArrow = false,
}: TooltipButtonProps) {
  const handleClick = (e: React.MouseEvent) => {
    if (onClick && !disabled) {
      onClick();
      e.preventDefault();
    }
  };

  const buttonClasses = cn(
    "hover:opacity-80",
    disabled && "opacity-50 cursor-not-allowed",
    className,
  );

  let content: ReactNode;

  if (navLinkTo && !disabled) {
    content = (
      <NavLink
        to={navLinkTo}
        onClick={handleClick}
        className={({ isActive }) =>
          cn(
            "hover:opacity-80",
            isActive ? "text-white" : "text-[#9099AC]",
            className,
          )
        }
        aria-label={ariaLabel}
        data-testid={testId}
      >
        {children}
      </NavLink>
    );
  } else if (navLinkTo && disabled) {
    content = (
      <button
        type="button"
        aria-label={ariaLabel}
        data-testid={testId}
        className={cn(
          "text-[#9099AC]",
          "opacity-50 cursor-not-allowed",
          className,
        )}
        disabled
      >
        {children}
      </button>
    );
  } else if (href && !disabled) {
    content = (
      <a
        href={href}
        target="_blank"
        rel="noreferrer noopener"
        className={cn("hover:opacity-80", className)}
        aria-label={ariaLabel}
        data-testid={testId}
      >
        {children}
      </a>
    );
  } else if (href && disabled) {
    content = (
      <button
        type="button"
        aria-label={ariaLabel}
        data-testid={testId}
        className={cn("opacity-50 cursor-not-allowed", className)}
        disabled
      >
        {children}
      </button>
    );
  } else {
    // Standard button with onClick
    content = (
      <button
        type="button"
        aria-label={ariaLabel}
        data-testid={testId}
        onClick={handleClick}
        className={buttonClasses}
        disabled={disabled}
      >
        {children}
      </button>
    );
  }

  return (
    <StyledTooltip
      content={tooltip}
      tooltipClassName={tooltipClassName}
      placement={placement}
      showArrow={showArrow}
    >
      {content}
    </StyledTooltip>
  );
}
