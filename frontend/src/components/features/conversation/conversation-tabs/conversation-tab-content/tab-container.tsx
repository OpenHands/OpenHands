import { ReactNode } from "react";

interface TabContainerProps {
  children: ReactNode;
}

export function TabContainer({ children }: TabContainerProps) {
  return (
    <div className="bg-[var(--color-surface-alt)] border border-[var(--color-border)] rounded-xl flex flex-col h-full w-full">
      {children}
    </div>
  );
}
