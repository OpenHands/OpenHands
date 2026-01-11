import { ReactNode } from "react";

interface TabContainerProps {
  children: ReactNode;
}

export function TabContainer({ children }: TabContainerProps) {
  return (
    <div className="bg-surface border border-stroke-alt rounded-xl flex flex-col h-full w-full">
      {children}
    </div>
  );
}
