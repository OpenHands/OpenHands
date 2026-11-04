import React from "react";
import { cn } from "#/utils/utils";

interface ChatInputGripProps {
  canResize: boolean;
  handleGripMouseDown: (e: React.MouseEvent) => void;
  handleGripTouchStart: (e: React.TouchEvent) => void;
}

export function ChatInputGrip({
  canResize,
  handleGripMouseDown,
  handleGripTouchStart,
}: ChatInputGripProps) {
  return (
    <div
      className={cn(
        "absolute top-0 left-0 w-full h-3 z-20",
        canResize ? "cursor-ns-resize select-none" : "pointer-events-none",
      )}
      id="resize-grip"
      onMouseDown={handleGripMouseDown}
      onTouchStart={handleGripTouchStart}
      aria-hidden
    />
  );
}
