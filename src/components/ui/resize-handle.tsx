import { useState } from "react";
import { cn } from "#/utils/utils";

interface ResizeHandleProps {
  onMouseDown: (e: React.MouseEvent) => void;
  className?: string;
  /** While the parent panel drag is active, keep the grip line highlighted. */
  isDragging?: boolean;
}

export function ResizeHandle({
  onMouseDown,
  className,
  isDragging = false,
}: ResizeHandleProps) {
  const [isHovering, setIsHovering] = useState(false);
  const lineActive = isDragging || isHovering;

  return (
    <div
      className={cn("relative z-10 w-0 shrink-0 self-stretch", className)}
      aria-hidden
    >
      <div
        className="absolute inset-y-0 left-1/2 w-3 min-w-[12px] -translate-x-1/2 cursor-ew-resize"
        onMouseDown={onMouseDown}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      />
      <div
        className={cn(
          "pointer-events-none absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors",
          lineActive ? "bg-white" : "bg-transparent",
        )}
        aria-hidden
      />
    </div>
  );
}
