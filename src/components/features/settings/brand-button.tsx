import { forwardRef } from "react";
import { cn } from "#/utils/utils";
import { formControlButtonClassName } from "#/utils/form-control-classes";

interface BrandButtonProps {
  testId?: string;
  name?: string;
  variant: "primary" | "secondary" | "tertiary" | "danger" | "ghost-danger";
  type: React.ButtonHTMLAttributes<HTMLButtonElement>["type"];
  isDisabled?: boolean;
  className?: string;
  onClick?: (event?: React.MouseEvent<HTMLButtonElement>) => void;
  startContent?: React.ReactNode;
  /** Accessible label for icon-only buttons */
  ariaLabel?: string;
  /** Indicates busy/loading state for screen readers */
  "aria-busy"?: boolean;
}

export const BrandButton = forwardRef<
  HTMLButtonElement,
  React.PropsWithChildren<BrandButtonProps>
>(function BrandButton(
  {
    testId,
    name,
    children,
    variant,
    type,
    isDisabled,
    className,
    onClick,
    startContent,
    ariaLabel,
    "aria-busy": ariaBusy,
  },
  ref,
) {
  return (
    <button
      ref={ref}
      name={name}
      data-testid={testId}
      disabled={isDisabled}
      // The type is already passed as a prop to the button component

      type={type}
      onClick={onClick}
      aria-label={ariaLabel}
      aria-busy={ariaBusy}
      className={cn(
        formControlButtonClassName,
        variant === "primary" &&
          "bg-primary text-[var(--oh-color-base)] hover:opacity-80 disabled:bg-[var(--oh-interactive-hover)] disabled:text-[var(--oh-muted)] disabled:opacity-100",
        variant === "secondary" &&
          "border border-[var(--oh-border)] bg-base-secondary text-white hover:bg-surface-raised",
        variant === "tertiary" &&
          "bg-[var(--oh-interactive-hover)] text-white hover:opacity-80",
        variant === "danger" && "bg-red-600 text-white hover:bg-red-700",
        variant === "ghost-danger" &&
          "h-auto min-h-0 bg-transparent px-0 text-red-600 underline hover:text-red-700 hover:no-underline font-normal",
        startContent && "flex items-center justify-center gap-2",
        className,
      )}
    >
      {startContent}
      {children}
    </button>
  );
});
