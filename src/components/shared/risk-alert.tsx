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
  // Currently, we are only supporting the high risk alert. If we use want to support other risk levels, we can add them here and use cva to create different variants of this component.
  if (severity === "high") {
    return (
      <div
        className={cn(
          "flex items-center gap-3.5 bg-[#4A0709] border border-[#FF0006] text-red-400 rounded-xl px-3.5 h-13 text-sm text-white",
          className,
        )}
      >
        {icon && <span className="">{icon}</span>}
        <span className="font-bold">{title}</span>
        <span className="font-normal">{content}</span>
      </div>
    );
  }

  return null;
}
