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
        "text-[var(--color-basic)] bg-[var(--color-base)]",
        isActive && "bg-[var(--color-surface-alt)] text-white",
        isActive
          ? "hover:text-white hover:bg-tertiary"
          : "hover:text-white hover:bg-[var(--color-base)]",
        isActive
          ? "focus-within:text-white"
          : "focus-within:text-[var(--color-basic)]",
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
