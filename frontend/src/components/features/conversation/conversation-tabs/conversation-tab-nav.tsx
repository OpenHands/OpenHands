import { ComponentType } from "react";
import { cn } from "#/utils/utils";

type ConversationTabNavProps = {
  tabValue: string;
  icon: ComponentType<{ className: string }>;
  onClick(): void;
  isActive?: boolean;
  label?: string;
  className?: string;
};

export function ConversationTabNav({
  tabValue,
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
      data-testid={`conversation-tab-${tabValue}`}
      className={cn(
        "flex items-center gap-2 rounded-lg cursor-pointer",
        "pl-1.5 pr-2 py-1",
        "transition-colors duration-150",
        "text-[#A1A1AA] bg-[#09090B]",
        isActive && "bg-[#6366F1] text-[#FAFAFA]",
        isActive
          ? "hover:text-[#FAFAFA] hover:bg-[#6366F1]/90"
          : "hover:text-[#FAFAFA] hover:bg-[#18181B]",
        isActive ? "focus-within:text-[#FAFAFA]" : "focus-within:text-[#A1A1AA]",
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
