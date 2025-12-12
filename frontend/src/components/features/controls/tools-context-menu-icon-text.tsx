import { cn } from "#/utils/utils";

interface ToolsContextMenuIconTextProps {
  icon: React.ReactNode;
  text: string;
  rightIcon?: React.ReactNode;
  className?: string;
}

export function ToolsContextMenuIconText({
  icon,
  text,
  rightIcon,
  className,
}: ToolsContextMenuIconTextProps) {
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
