import {
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
  useState,
} from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { dropdownMenuListClassName } from "#/utils/dropdown-classes";
import { I18nKey } from "#/i18n/declaration";
import { ConversationNameContextMenuIconText } from "#/components/features/conversation/conversation-name-context-menu-icon-text";
import EditIcon from "#/icons/u-edit.svg?react";
import CheckCircleIcon from "#/icons/u-check-circle.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";

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

function MenuItem({
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

interface MetaProfileActionsMenuProps {
  onEdit: () => void;
  onSetActive: () => void;
  onDelete: () => void;
  isActive: boolean;
  isActivating: boolean;
  onClose: () => void;
  /**
   * Element the menu should anchor against. When provided, the menu renders
   * into a portal at the document body using fixed positioning so it cannot be
   * clipped by ancestors with `overflow: auto/hidden` (e.g. the settings
   * `<main>` scroll container).
   */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

export function MetaProfileActionsMenu({
  onEdit,
  onSetActive,
  onDelete,
  isActive,
  isActivating,
  onClose,
  anchorRef,
}: MetaProfileActionsMenuProps) {
  const { t } = useTranslation("openhands");
  const menuRef = useRef<HTMLDivElement>(null);
  const menuItemsRef = useRef<(HTMLButtonElement | null)[]>([]);

  const anchorElement = anchorRef?.current ?? null;
  const [portalStyle, setPortalStyle] = useState<React.CSSProperties>();

  useLayoutEffect(() => {
    if (!anchorElement) return undefined;

    const updatePosition = () => {
      const rect = anchorElement.getBoundingClientRect();
      if (!rect) return;
      const gap = 8;
      setPortalStyle({
        position: "fixed",
        zIndex: 9999,
        top: rect.bottom + gap,
        right: window.innerWidth - rect.right,
        width: "max-content",
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [anchorElement]);

  // Focus first item when menu opens
  useEffect(() => {
    menuItemsRef.current[0]?.focus();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (menuRef.current && !menuRef.current.contains(target)) {
        onClose();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  const handleAction = (action: () => void) => {
    action();
    onClose();
  };

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, currentIndex: number) => {
      if (e.key === "Tab") {
        onClose();
        return;
      }
      const itemCount = menuItemsRef.current.filter(Boolean).length;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        const nextIndex = (currentIndex + 1) % itemCount;
        menuItemsRef.current[nextIndex]?.focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const prevIndex = (currentIndex - 1 + itemCount) % itemCount;
        menuItemsRef.current[prevIndex]?.focus();
      }
    },
    [onClose],
  );

  const setActiveDisabled = isActive || isActivating;
  const isPortaled = Boolean(anchorElement);

  const menu = (
    <div
      ref={menuRef}
      className={cn(
        "absolute right-0 top-full z-10 mt-2 w-[160px] rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 shadow-lg",
        dropdownMenuListClassName,
        isPortaled &&
          "!static !top-auto !bottom-auto !left-auto !right-auto !mt-0",
      )}
      role="menu"
      aria-orientation="vertical"
      data-testid="meta-profile-actions-menu"
    >
      <MenuItem
        index={0}
        icon={<EditIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$EDIT)}
        onClick={() => handleAction(onEdit)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="meta-profile-edit"
      />
      <MenuItem
        index={1}
        icon={<CheckCircleIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$META_PROFILE_ACTIVATE)}
        onClick={() => handleAction(onSetActive)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        disabled={setActiveDisabled}
        testId="meta-profile-set-active"
      />
      <MenuItem
        index={2}
        icon={<DeleteIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$DELETE)}
        onClick={() => handleAction(onDelete)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="meta-profile-delete"
      />
    </div>
  );

  if (isPortaled) {
    if (typeof document === "undefined" || !portalStyle) {
      return null;
    }
    return ReactDOM.createPortal(
      // portal position computed from DOM bounding rect at runtime
      <div style={portalStyle}>{menu}</div>,
      document.body,
    );
  }

  return menu;
}
