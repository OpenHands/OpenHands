import { ReactNode } from "react";
import { cn } from "#/utils/utils";

interface MetricRowProps {
  label: ReactNode;
  value: ReactNode;
  labelClassName?: string;
  valueClassName?: string;
  showBorder?: boolean;
}

export function MetricRow({
  label,
  value,
  labelClassName = "",
  valueClassName = "font-semibold",
  showBorder = true,
}: MetricRowProps) {
  return (
    <div
      className={cn(
        "flex justify-between items-center pb-2",
        showBorder && "border-b border-[var(--oh-border-subtle)]",
      )}
    >
      <span className={labelClassName}>{label}</span>
      <span className={valueClassName}>{value}</span>
    </div>
  );
}
