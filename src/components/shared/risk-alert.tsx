import { ReactNode } from "react";
import { cn } from "#/utils/utils";

type RiskAlertSeverity = "high" | "medium" | "low";

interface RiskAlertProps {
  className?: string;
  content: ReactNode;
  icon?: ReactNode;
  severity: RiskAlertSeverity;
  title: string;
}

const SEVERITY_VARIANT_CLASSES: Partial<Record<RiskAlertSeverity, string>> = {
  high: "bg-danger/15 border border-danger text-contrast dark:border-[#FF0006] dark:bg-[#4A0709] dark:text-white",
  medium:
    "bg-warning/15 border border-warning text-contrast dark:border-[#D97706] dark:bg-[#3D2000] dark:text-white",
};

export function RiskAlert({
  className,
  content,
  icon,
  severity,
  title,
}: RiskAlertProps) {
  const variantClasses = SEVERITY_VARIANT_CLASSES[severity];
  if (!variantClasses) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex items-center gap-3.5 rounded-xl px-3.5 min-h-13 py-2 text-sm",
        variantClasses,
        className,
      )}
    >
      {icon && <span className="">{icon}</span>}
      <span className="font-bold">{title}</span>
      <span className="font-normal">{content}</span>
    </div>
  );
}
