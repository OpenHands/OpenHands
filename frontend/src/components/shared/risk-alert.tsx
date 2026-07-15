import { ReactNode } from "react";
import { cn } from "#/utils/utils";

interface RiskAlertProps {
  className?: string;
  content: ReactNode;
  icon?: ReactNode;
  severity: "high" | "medium" | "low";
  title: string;
}

export function RiskAlert({
  className,
  content,
  icon,
  severity,
  title,
}: RiskAlertProps) {
  if (severity === "high") {
    return (
      <div
        className={cn(
          "flex items-center gap-3.5 bg-[#4A0709] border border-[#FF0006] rounded-xl px-3.5 h-13 text-sm text-white",
          className,
        )}
      >
        {icon && <span className="text-red-400">{icon}</span>}
        <span className="font-bold">{title}</span>
        <span className="font-normal">{content}</span>
      </div>
    );
  }

  if (severity === "medium") {
    return (
      <div
        className={cn(
          "flex items-center gap-3.5 bg-[#3a2a06] border border-amber-500 rounded-xl px-3.5 h-13 text-sm text-white",
          className,
        )}
      >
        {icon && <span className="text-amber-400">{icon}</span>}
        <span className="font-bold">{title}</span>
        <span className="font-normal">{content}</span>
      </div>
    );
  }

  return null;
}
