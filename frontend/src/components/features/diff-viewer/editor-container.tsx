import React from "react";

interface EditorContainerProps {
  height: number;
  children: React.ReactNode;
}

export function EditorContainer({ height, children }: EditorContainerProps) {
  return (
    <div
      data-testid="editor-container"
      className="w-full border border-neutral-600 overflow-hidden h-[var(--editor-height)] rounded-bl-xl rounded-br-xl"
      style={{ "--editor-height": `${height}px` } as React.CSSProperties}
    >
      {children}
    </div>
  );
}
