import { cn } from "#/utils/utils";
import { ConversationNameContextMenuIconText } from "#/components/features/conversation/conversation-name-context-menu-icon-text";

interface MenuItemProps {
  index: number;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  onKeyDown: (e: React.KeyboardEvent, index: number) => void;
  menuItemsRef: React.MutableRefObject<(HTMLButtonElement | null)[]>;
  disabled?: boolean;
  testId: string;
}

export function MenuItem({
  index,
  icon,
  label,
  onClick,
  onKeyDown,
  menuItemsRef,
  disabled,
  testId,
}: MenuItemProps) {
  return (
    <button
      ref={(el) => {
        // eslint-disable-next-line no-param-reassign
        menuItemsRef.current[index] = el;
      }}
      type="button"
      onClick={onClick}
      onKeyDown={(e) => onKeyDown(e, index)}
      disabled={disabled}
      className={cn(
        "group w-full cursor-pointer rounded px-2 py-2 text-start text-nowrap text-sm font-normal",
        "text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)]",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent",
      )}
      role="menuitem"
      data-testid={testId}
    >
      <ConversationNameContextMenuIconText icon={icon} text={label} />
    </button>
  );
}
