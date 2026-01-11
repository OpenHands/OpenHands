import { cn } from "#/utils/utils";

interface ContextMenuSeparatorProps {
  className?: string;
  testId?: string;
}

export function ContextMenuSeparator({
  className,
  testId,
}: ContextMenuSeparatorProps) {
  return (
    <div
      data-testid={testId}
      className={cn("w-full h-px bg-stroke-alt", className)}
    />
  );
}
