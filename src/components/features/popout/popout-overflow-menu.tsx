import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import type { Popout } from "#/stores/popout-store";
import {
  dropdownMenuListClassName,
  dropdownMenuRowClassName,
  dropdownMenuViewportScrollClassName,
} from "#/utils/dropdown-classes";
import { PopoutStatusDot } from "./popout-status-dot";

interface PopoutOverflowMenuProps {
  hiddenPopouts: Popout[];
  onSelect: (conversationId: string) => void;
}

/** Circular overflow count with an accessible selector for hidden popouts. */
export function PopoutOverflowMenu({
  hiddenPopouts,
  onSelect,
}: PopoutOverflowMenuProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const menuId = React.useId();
  const containerRef = useClickOutsideElement<HTMLDivElement>(() =>
    setOpen(false),
  );

  React.useEffect(() => {
    if (!open) return;
    const firstItem =
      containerRef.current?.querySelector<HTMLButtonElement>(
        '[role="menuitem"]',
      );
    firstItem?.focus();
  }, [containerRef, open]);

  if (hiddenPopouts.length === 0) return null;

  const closeAndRestoreFocus = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };

  const handleMenuKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeAndRestoreFocus();
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;

    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>(
        '[role="menuitem"]',
      ),
    );
    if (items.length === 0) return;
    const currentIndex = items.indexOf(
      document.activeElement as HTMLButtonElement,
    );
    const delta = event.key === "ArrowDown" ? 1 : -1;
    const startIndex =
      currentIndex === -1 ? (delta > 0 ? -1 : 0) : currentIndex;
    const nextIndex = (startIndex + delta + items.length) % items.length;
    event.preventDefault();
    items[nextIndex]?.focus();
  };

  return (
    <div ref={containerRef} className="relative shrink-0">
      <button
        ref={triggerRef}
        type="button"
        data-testid="popout-overflow-trigger"
        aria-label={t(I18nKey.POPOUT$SHOW_HIDDEN, {
          count: hiddenPopouts.length,
        })}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((current) => !current)}
        className="flex size-9 items-center justify-center rounded-full border border-[var(--oh-border)] bg-[var(--oh-surface)] text-xs font-medium text-[var(--oh-foreground)] shadow-lg transition-colors hover:bg-[var(--oh-surface-raised)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--oh-interactive-accent)]"
      >
        {hiddenPopouts.length}
      </button>

      {open ? (
        <div
          id={menuId}
          role="menu"
          aria-label={t(I18nKey.POPOUT$HIDDEN_MENU)}
          tabIndex={-1}
          data-testid="popout-overflow-menu"
          onKeyDown={handleMenuKeyDown}
          className={`absolute bottom-full right-0 mb-2 w-64 rounded-md border border-[var(--oh-border-subtle)] bg-tertiary p-1 text-[var(--oh-foreground)] shadow-lg ${dropdownMenuListClassName} ${dropdownMenuViewportScrollClassName}`}
        >
          {hiddenPopouts.map((popout) => (
            <button
              key={popout.conversationId}
              type="button"
              role="menuitem"
              data-testid={`popout-overflow-item-${popout.conversationId}`}
              onClick={() => {
                onSelect(popout.conversationId);
                closeAndRestoreFocus();
              }}
              className={dropdownMenuRowClassName}
            >
              <PopoutStatusDot
                conversationId={popout.conversationId}
                showTooltip={false}
              />
              <span className="min-w-0 flex-1 truncate text-left">
                {popout.title}
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
