import { Check, Grid2x2 } from "lucide-react";
import { dropdownMenuRowIconWrapperClassName } from "#/utils/dropdown-classes";

export function ViewMenuItemContent({
  icon: Icon,
  label,
  isSelected,
}: {
  icon: typeof Grid2x2;
  label: string;
  isSelected: boolean;
}) {
  return (
    <span className="flex min-w-0 w-full items-center gap-2">
      <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
        <Icon />
      </span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {isSelected ? <Check className="size-4 shrink-0" aria-hidden /> : null}
    </span>
  );
}
