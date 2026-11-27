import type { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";

export function BackNavButtonContent({ children }: { children: ReactNode }) {
  return (
    <>
      <ArrowLeft size={20} aria-hidden />
      {children}
    </>
  );
}
