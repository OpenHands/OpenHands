import { cn, getStatusClassName } from "#/utils/utils";

interface StatusBadgeProps {
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "text-xs px-2 py-1 rounded uppercase font-semibold",
        getStatusClassName(status),
      )}
    >
      {status.replace("_", " ")}
    </span>
  );
}
