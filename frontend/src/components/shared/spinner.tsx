import { LoaderCircle } from "lucide-react";
import { cn } from "#/utils/utils";

export type SpinnerSize = "xs" | "sm" | "md" | "lg" | "xl";

export interface SpinnerProps {
  size?: SpinnerSize;
  className?: string;
}

const legacySizeMap: Record<string, SpinnerSize> = {
  small: "sm",
  medium: "md",
  large: "lg",
};

const tailwindSizeMap: Record<SpinnerSize, string> = {
  xs: "w-3 h-3",
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
  xl: "w-16 h-16",
};

export function Spinner({ size = "md", className }: SpinnerProps) {
  const resolvedSize = legacySizeMap[size] ?? size;
  return (
    <LoaderCircle
      className={cn(
        "animate-spin text-[inherit]",
        tailwindSizeMap[resolvedSize],
        className,
      )}
    />
  );
}

export interface LoadingContainerProps {
  children?: React.ReactNode;
  className?: string;
  spinnerClassName?: string;
  size?: "xs" | "sm" | "md" | "lg" | "xl";
}

export function LoadingContainer({
  children,
  className,
  spinnerClassName,
  size = "md",
}: LoadingContainerProps) {
  return (
    <div className={cn("flex items-center justify-center", className)}>
      <Spinner size={size} className={spinnerClassName} />
      {children}
    </div>
  );
}
