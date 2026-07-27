import React from "react";
import { cn } from "#/utils/utils";

interface ChatInputGripProps {
  gripRef: React.RefObject<HTMLDivElement | null>;
  isGripVisible: boolean;
  isGripDragging: boolean;
  handleTopEdgeClick: (e: React.MouseEvent) => void;
  handleGripMouseDown: (e: React.MouseEvent) => void;
  handleGripTouchStart: (e: React.TouchEvent) => void;
}

export function ChatInputGrip({
  gripRef,
  isGripVisible,
  isGripDragging,
  handleTopEdgeClick,
  handleGripMouseDown,
  handleGripTouchStart,
}: ChatInputGripProps) {
  return (
    <div
      className="absolute top-0 left-0 w-full h-3 z-20 group"
      id="resize-grip"
      onClick={handleTopEdgeClick}
    >
      {/* Resize hit target; 1px line is drawn at top-0 (flush with chat input box) */}
      <div
        className="absolute inset-0 z-[1] cursor-ns-resize select-none"
        onMouseDown={handleGripMouseDown}
        onTouchStart={handleGripTouchStart}
        aria-hidden
      />
      <div
        ref={gripRef}
        className={cn(
          "pointer-events-none absolute top-0 left-0 w-full h-px bg-white z-[2] transition-opacity duration-200",
          isGripVisible || isGripDragging
            ? "opacity-100"
            : "opacity-0 group-hover:opacity-100",
        )}
      />
    </div>
  );
}
