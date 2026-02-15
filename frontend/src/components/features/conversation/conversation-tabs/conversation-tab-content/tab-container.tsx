import { ReactNode } from "react";

interface TabContainerProps {
  children: ReactNode;
}

export function TabContainer({ children }: TabContainerProps) {
  return (
    <div className="bg-[#18181B] border border-[#27272A] rounded-lg flex flex-col h-full w-full">
      {children}
    </div>
  );
}
