import { cn } from "#/utils/utils";

interface CreditsChipProps {
  testId?: string;
  className?: string;
}

/**
 * Chip component for displaying credits amount
 * Uses yellow background with black text for visibility
 */
export function CreditsChip({
  children,
  testId,
  className,
}: React.PropsWithChildren<CreditsChipProps>) {
  return (
    <div
      data-testid={testId}
      data-openhands-chip
      style={{ minWidth: "100px" }}
      className={cn(
        "bg-[#FFE165] px-4 rounded-[100px] text-black text-lg text-center font-semibold",
        className,
      )}
    >
      {children}
    </div>
  );
}
