import { Spinner as HeroUISpinner } from "@heroui/react";
import { cn } from "#/utils/utils";

/**
 * Spinner size variants following the design system
 */
export type SpinnerSize = "sm" | "md" | "lg" | "xl";

/**
 * Props for the unified Spinner component
 */
interface SpinnerProps {
  /**
   * The size of the spinner
   * @default "md"
   */
  size?: SpinnerSize;
  /**
   * Optional label text to display next to the spinner
   */
  label?: string;
  /**
   * Additional CSS classes for the container
   */
  className?: string;
  /**
   * Additional CSS classes for the spinner element
   */
  spinnerClassName?: string;
  /**
   * Data test id for testing
   */
  dataTestId?: string;
  /**
   * Color of the spinner
   * @default "current"
   */
  color?: "current" | "primary" | "secondary" | "success" | "warning" | "danger";
}

/**
 * Map our size values to HeroUI spinner sizes
 */
const sizeToHeroUISize: Record<SpinnerSize, "sm" | "md" | "lg"> = {
  sm: "sm",
  md: "md",
  lg: "lg",
  xl: "lg",
};

/**
 * Map our size values to Tailwind classes for the XL variant
 */
const sizeToClassName: Record<SpinnerSize, string> = {
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
  xl: "w-16 h-16",
};

/**
 * Unified Spinner component
 *
 * This component consolidates all spinner implementations into a single,
 * reusable component with consistent sizing and styling.
 *
 * @example
 * ```tsx
 * // Simple spinner
 * <Spinner />
 *
 * // With label
 * <Spinner label="Loading..." />
 *
 * // Large size with custom class
 * <Spinner size="lg" className="text-primary" />
 * ```
 */
export function Spinner({
  size = "md",
  label,
  className,
  spinnerClassName,
  dataTestId = "spinner",
  color = "current",
}: SpinnerProps) {
  const heroUISize = sizeToHeroUISize[size];
  const sizeClass = size === "xl" ? sizeToClassName[size] : undefined;

  return (
    <div
      data-testid={dataTestId}
      className={cn(
        "flex items-center gap-2",
        label ? "flex-row" : "justify-center",
        className
      )}
    >
      {size === "xl" ? (
        // For XL size, use custom styling since HeroUI only goes up to lg
        <div
          className={cn(
            "animate-spin rounded-full border-2 border-current border-t-transparent",
            sizeClass,
            spinnerClassName
          )}
        />
      ) : (
        <HeroUISpinner
          size={heroUISize}
          color={color}
          className={spinnerClassName}
        />
      )}
      {label && (
        <span className="text-sm text-current">{label}</span>
      )}
    </div>
  );
}

export default Spinner;

