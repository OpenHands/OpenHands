import { ReactNode } from "react";

interface TabContainerProps {
  children: ReactNode;
}

export function TabContainer({ children }: TabContainerProps) {
  return <div className="flex flex-col h-full w-full">{children}</div>;
}
