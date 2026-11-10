import React from "react";
import { cn } from "#/utils/utils";

interface EditorContainerProps {
  height: number;
  children: React.ReactNode;
  className?: string;
}

export function EditorContainer({
  height,
  children,
  className,
}: EditorContainerProps) {
  return (
    <div
      data-testid="editor-container"
      className={cn(
        "w-full border-b border-border overflow-hidden h-(--editor-height)",
        className,
      )}
      style={{ "--editor-height": `${height}px` } as React.CSSProperties}
    >
      {children}
    </div>
  );
}
