import { cn } from "#/utils/utils";
import { dropdownMenuRowIconWrapperClassName } from "#/utils/dropdown-classes";

interface KebabMenuItemContentProps {
  icon: React.ReactNode;
  label: string;
}

export function KebabMenuItemContent({
  icon,
  label,
}: KebabMenuItemContentProps) {
  return (
    <span className="flex min-w-0 w-full items-center gap-2">
      <span
        className={cn("[&_svg]:size-4", dropdownMenuRowIconWrapperClassName)}
        aria-hidden
      >
        {icon}
      </span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </span>
  );
}
