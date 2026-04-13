import { Spinner, SpinnerSize } from "#/components/shared/spinner";
import { cn } from "#/utils/utils";

interface LoaderProps {
  size?: "small" | "medium" | "large";
  className?: string;
}

const sizeMap: Record<"small" | "medium" | "large", SpinnerSize> = {
  small: "xs",
  medium: "sm",
  large: "md",
};

export function Loader({ size = "medium", className }: LoaderProps) {
  return (
    <Spinner
      size={sizeMap[size]}
      testId="loader"
      className={cn("flex items-center justify-center", className)}
    />
  );
}
