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
      className={cn("w-full h-[1px] bg-[#525252]", className)}
    />
  );
}
