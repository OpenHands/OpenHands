import { Spinner } from "./spinner";

const sizeMap: Record<string, "xs" | "sm" | "md" | "lg" | "xl"> = {
  small: "sm",
  medium: "md",
  large: "lg",
};

interface LoadingSpinnerProps {
  size?: "small" | "medium" | "large" | "xs" | "sm" | "md" | "lg" | "xl";
  className?: string;
}

export function LoadingSpinner({
  size = "md",
  className,
}: LoadingSpinnerProps) {
  return (
    <div data-testid="loading-spinner" className={className}>
      <Spinner size={sizeMap[size] ?? size} />
    </div>
  );
}
