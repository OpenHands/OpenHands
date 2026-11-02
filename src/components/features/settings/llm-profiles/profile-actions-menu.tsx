import {
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
  useState,
} from "react";
import ReactDOM from "react-dom";
import { useTranslation } from "react-i18next";
import { TextCursor, Copy } from "lucide-react";
import { cn } from "#/utils/utils";
import { dropdownMenuListClassName } from "#/utils/dropdown-classes";
import { I18nKey } from "#/i18n/declaration";
import EditIcon from "#/icons/u-edit.svg?react";
import CheckCircleIcon from "#/icons/u-check-circle.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import { MenuItem } from "./profile-actions-menu-item";

interface ProfileActionsMenuProps {
  onEdit: () => void;
  onRename: () => void;
  onDuplicate: () => void;
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

// A natively disabled button ignores focus(), so keyboard navigation has to
// skip it rather than stop on it.
const getEnabledItems = (items: (HTMLButtonElement | null)[]) =>
  items.filter((item): item is HTMLButtonElement =>
    Boolean(item && !item.disabled),
  );

export function ProfileActionsMenu({
  onEdit,
  onRename,
  onDuplicate,
  onSetActive,
  onDelete,
  isActive,
  isActivating,
  onClose,
  anchorRef,
}: ProfileActionsMenuProps) {
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

  // The anchored menu renders nothing until its position is measured, so wait
  // for the items to exist before focusing the first enabled one.
  const isMenuRendered = !anchorElement || portalStyle !== undefined;
  useEffect(() => {
    if (isMenuRendered) getEnabledItems(menuItemsRef.current)[0]?.focus();
  }, [isMenuRendered]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (anchorElement?.contains(target)) return;
      onClose();
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (menuRef.current?.contains(document.activeElement)) {
          anchorElement?.focus();
        }
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [anchorElement, onClose]);

  const handleAction = (action: () => void) => {
    action();
    onClose();
  };

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, currentIndex: number) => {
      if (e.key === "Tab") {
        // The anchored menu is portaled to the end of <body>; move focus back
        // to its trigger so Tab continues from the row, not the page end.
        anchorElement?.focus();
        onClose();
        return;
      }
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const enabledItems = getEnabledItems(menuItemsRef.current);
      const position = enabledItems.findIndex(
        (item) => item === menuItemsRef.current[currentIndex],
      );
      const step = e.key === "ArrowDown" ? 1 : -1;
      const target =
        (position + step + enabledItems.length) % enabledItems.length;
      enabledItems[target]?.focus();
    },
    [anchorElement, onClose],
  );

  const setActiveDisabled = isActive || isActivating;
  const isPortaled = Boolean(anchorElement);

  const menu = (
    <div
      ref={menuRef}
      className={cn(
        "absolute right-0 top-full z-10 mt-2 w-40 rounded-md border border-border-subtle bg-tertiary px-1 py-1 shadow-lg",
        dropdownMenuListClassName,
        isPortaled &&
          "!static !top-auto !bottom-auto !left-auto !right-auto !mt-0",
      )}
      role="menu"
      aria-orientation="vertical"
      data-testid="profile-actions-menu"
    >
      <MenuItem
        index={0}
        icon={<EditIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$PROFILE_EDIT)}
        onClick={() => handleAction(onEdit)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="profile-edit"
      />
      <MenuItem
        index={1}
        icon={<TextCursor aria-hidden className="size-4" strokeWidth={2} />}
        label={t(I18nKey.BUTTON$RENAME)}
        onClick={() => handleAction(onRename)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="profile-rename"
      />
      <MenuItem
        index={2}
        icon={<Copy aria-hidden className="size-4" strokeWidth={2} />}
        label={t(I18nKey.BUTTON$DUPLICATE)}
        onClick={() => handleAction(onDuplicate)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="profile-duplicate"
      />
      <MenuItem
        index={3}
        icon={<CheckCircleIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$PROFILE_SET_DEFAULT)}
        onClick={() => handleAction(onSetActive)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        disabled={setActiveDisabled}
        testId="profile-set-active"
      />
      {/* The active profile can be deleted: useEnsureActiveProfile then promotes
          another remaining profile so a profile is always active in local mode. */}
      <MenuItem
        index={4}
        icon={<DeleteIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$DELETE)}
        onClick={() => handleAction(onDelete)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="profile-delete"
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
