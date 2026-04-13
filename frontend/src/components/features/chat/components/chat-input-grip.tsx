import React from "react";
import { cn } from "#/utils/utils";

interface ChatInputGripProps {
  gripRef: React.RefObject<HTMLDivElement | null>;
  isGripVisible: boolean;
  handleTopEdgeClick: (e: React.MouseEvent) => void;
  handleGripMouseDown: (e: React.MouseEvent) => void;
  handleGripTouchStart: (e: React.TouchEvent) => void;
  isEnabled?: boolean;
}

export function ChatInputGrip({
  gripRef,
  isGripVisible,
  handleTopEdgeClick,
  handleGripMouseDown,
  handleGripTouchStart,
  isEnabled = true,
}: ChatInputGripProps) {
  if (!isEnabled) {
    return null;
  }

  return (
    <div
      className="absolute -top-[12px] left-0 w-full h-6 lg:h-3 z-20 group"
      id="resize-grip"
      onClick={handleTopEdgeClick}
    >
      {/* Resize Grip - appears on hover of top edge area, when dragging, or when clicked */}
      <div
        ref={gripRef}
        className={cn(
          "absolute top-[4px] left-0 w-full h-[3px] bg-white cursor-ns-resize z-10 transition-opacity duration-200",
          isGripVisible ? "opacity-100" : "opacity-0 group-hover:opacity-100",
        )}
        onMouseDown={handleGripMouseDown}
        onTouchStart={handleGripTouchStart}
        style={{ userSelect: "none" }}
      />
    </div>
  );
}
