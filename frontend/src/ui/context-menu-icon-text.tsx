import { cn } from "#/utils/utils";

interface ContextMenuIconTextProps {
  icon: React.ReactNode;
  text: string;
  rightIcon?: React.ReactNode;
  className?: string;
}

export function ContextMenuIconText({
  icon,
  text,
  rightIcon,
  className,
}: ContextMenuIconTextProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between p-2 hover:bg-[#5C5D62] rounded",
        className,
      )}
    >
      <div className="flex items-center gap-2">
        {icon}
        {text}
      </div>
      {rightIcon && <div className="flex items-center">{rightIcon}</div>}
    </div>
  );
}
