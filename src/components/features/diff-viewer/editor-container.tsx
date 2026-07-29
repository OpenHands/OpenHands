import React from "react";

interface EditorContainerProps {
  height: number;
  children: React.ReactNode;
}

export function EditorContainer({ height, children }: EditorContainerProps) {
  return (
    <div
      data-testid="editor-container"
      className="w-full border-b border-[var(--oh-border)] overflow-hidden h-[var(--editor-height)]"
      // CSS custom property plumbed through for h-[var(--editor-height)] above
      style={{ "--editor-height": `${height}px` } as React.CSSProperties}
    >
      {children}
    </div>
  );
}
