import { ReactNode } from "react";

interface TabContainerProps {
  children: ReactNode;
}

export function TabContainer({ children }: TabContainerProps) {
  return (
    <div className="bg-base-secondary border border-[#525252] rounded-xl flex flex-col h-full w-full">
      {children}
    </div>
  );
}
