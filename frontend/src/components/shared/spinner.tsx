import { cn } from "#/utils/utils";

export type SpinnerSize = "sm" | "md" | "lg" | "xl";

interface SpinnerProps {
  size?: SpinnerSize;
  className?: string;
  label?: string;
}

const sizeClasses: Record<SpinnerSize, string> = {
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
  xl: "w-16 h-16",
};

const strokeWidths: Record<SpinnerSize, number> = {
  sm: 2,
  md: 2,
  lg: 3,
  xl: 4,
};

export function Spinner({ size = "md", className, label }: SpinnerProps) {
  const sizeClass = sizeClasses[size];
  const strokeWidth = strokeWidths[size];

  return (
    <div
      data-testid="spinner"
      className={cn(
        "flex flex-col items-center justify-center gap-2",
        className,
      )}
    >
      <svg
        className={cn("animate-spin", sizeClass)}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth={strokeWidth}
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      {label && <span className="text-sm text-gray-600">{label}</span>}
    </div>
  );
}
