import React from "react";
import { cn } from "#/utils/utils";

interface ChatInputGripProps {
  gripRef: React.RefObject<HTMLDivElement | null>;
  isGripVisible: boolean;
  isGripDragging: boolean;
  canResize: boolean;
  handleTopEdgeClick: (e: React.MouseEvent) => void;
  handleGripMouseDown: (e: React.MouseEvent) => void;
  handleGripTouchStart: (e: React.TouchEvent) => void;
}

export function ChatInputGrip({
  gripRef,
  isGripVisible,
  isGripDragging,
  canResize,
  handleTopEdgeClick,
  handleGripMouseDown,
  handleGripTouchStart,
}: ChatInputGripProps) {
  return (
    <div
      className={cn(
        "absolute top-0 left-0 w-full h-3 z-20 group",
        !canResize && "pointer-events-none",
      )}
      id="resize-grip"
      onClick={canResize ? handleTopEdgeClick : undefined}
    >
      {/* Resize hit target; 1px line is drawn at top-0 (flush with chat input box) */}
      {canResize && (
        <div
          className="absolute inset-0 z-[1] cursor-ns-resize select-none"
          onMouseDown={handleGripMouseDown}
          onTouchStart={handleGripTouchStart}
          aria-hidden
        />
      )}
      <div
        ref={gripRef}
        className={cn(
          "pointer-events-none absolute top-0 left-0 w-full h-px bg-white z-[2] transition-opacity duration-200",
          !canResize && "opacity-0",
          canResize &&
            (isGripVisible || isGripDragging
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100"),
        )}
      />
    </div>
  );
}
