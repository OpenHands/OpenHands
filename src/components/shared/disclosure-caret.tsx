import { cn } from "#/utils/utils";

/** Small solid triangle: points right when closed, down when open. */
export function DisclosureCaret({
  open,
  className,
}: {
  open: boolean;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 10 10"
      aria-hidden
      className={cn(
        "h-2.5 w-2.5 flex-shrink-0 fill-muted transition-transform",
        open && "rotate-90",
        className,
      )}
    >
      <path d="M3.5 1.5 L8 5 L3.5 8.5 Z" />
    </svg>
  );
}
