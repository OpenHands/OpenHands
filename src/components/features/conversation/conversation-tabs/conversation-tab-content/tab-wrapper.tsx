import { ReactNode } from "react";

interface TabWrapperProps {
  children: ReactNode;
}

export function TabWrapper({ children }: TabWrapperProps) {
  return <div className="absolute inset-0">{children}</div>;
}
