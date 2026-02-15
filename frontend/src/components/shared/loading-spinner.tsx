import { cn } from "#/utils/utils";

interface LoadingSpinnerProps {
  size: "small" | "large";
}

export function LoadingSpinner({ size }: LoadingSpinnerProps) {
  const sizeStyle =
    size === "small" ? "w-[25px] h-[25px]" : "w-[50px] h-[50px]";

  return (
    <div data-testid="loading-spinner" className={cn("relative", sizeStyle)}>
      <div
        className={cn(
          "rounded-full border-[3px] border-[#27272A] absolute",
          sizeStyle,
        )}
      />
      <div
        className={cn(
          "rounded-full border-[3px] border-transparent border-t-[#6366F1] absolute animate-spin",
          sizeStyle,
        )}
      />
    </div>
  );
}
