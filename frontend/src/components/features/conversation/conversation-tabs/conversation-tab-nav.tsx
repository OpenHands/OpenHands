import { ComponentType } from "react";
import { cn } from "#/utils/utils";

type ConversationTabNavProps = {
  icon: ComponentType<{ className: string }>;
  onClick(): void;
  isActive?: boolean;
  label?: string;
  className?: string;
};

export function ConversationTabNav({
  icon: Icon,
  onClick,
  isActive,
  label,
  className,
}: ConversationTabNavProps) {
  return (
    <button
      type="button"
      onClick={() => {
        onClick();
      }}
      className={cn(
        "flex items-center gap-2 rounded-md cursor-pointer",
        "pl-1.5 pr-2 py-1",
        "text-basic bg-base",
        isActive && "bg-base-secondary text-content",
        isActive
          ? "hover:text-content hover:bg-tertiary"
          : "hover:text-content hover:bg-base",
        isActive ? "focus-within:text-content" : "focus-within:text-[#9299AA]",
        className,
      )}
    >
      <Icon className={cn("w-5 h-5 text-inherit flex-shrink-0")} />
      {isActive && label && (
        <span className="text-sm font-medium whitespace-nowrap">{label}</span>
      )}
    </button>
  );
}
