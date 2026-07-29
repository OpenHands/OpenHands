import { useState, useEffect, useLayoutEffect, useRef } from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import KebabVerticalIcon from "#/icons/kebab-vertical.svg?react";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import { automationIconActionButtonClassName } from "./automation-action-button-classes";
import { KebabMenuItemContent } from "./kebab-menu-item-content";

export interface KebabMenuItem {
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}

interface KebabMenuProps {
  items: KebabMenuItem[];
}

export function KebabMenu({ items }: KebabMenuProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const [portalStyle, setPortalStyle] = useState<React.CSSProperties>();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return undefined;

    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) return;

      const gap = 4;
      setPortalStyle({
        position: "fixed",
        zIndex: 9999,
        top: rect.bottom + gap,
        right: window.innerWidth - rect.right,
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        triggerRef.current?.contains(target) ||
        menuRef.current?.contains(target)
      ) {
        return;
      }
      setOpen(false);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [open]);

  const menu =
    open && portalStyle ? (
      <ContextMenu ref={menuRef} theme="popover" className="min-w-[10rem]">
        {items.map((item) => (
          <li key={item.label}>
            <ContextMenuListItem
              onClick={(event) => {
                event.stopPropagation();
                item.onClick();
                setOpen(false);
              }}
              isDisabled={item.disabled}
              className="group"
            >
              <KebabMenuItemContent icon={item.icon} label={item.label} />
            </ContextMenuListItem>
          </li>
        ))}
      </ContextMenu>
    ) : null;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((current) => !current);
        }}
        className={automationIconActionButtonClassName}
        aria-label={t(I18nKey.AUTOMATIONS$ACTIONS_MENU)}
        aria-expanded={open}
        aria-haspopup="menu"
      >
        <KebabVerticalIcon className="size-4" />
      </button>

      {open && portalStyle && typeof document !== "undefined"
        ? ReactDOM.createPortal(
            <div style={portalStyle}>{menu}</div>,
            document.body,
          )
        : null}
    </>
  );
}
