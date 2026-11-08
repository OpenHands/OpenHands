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
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import AddModelsIcon from "#/icons/u-plus.svg?react";
import { MenuItem } from "./profile-actions-menu-item";

interface ProviderConnectionActionsMenuProps {
  onAddModels: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onClose: () => void;
  /**
   * Element the menu should anchor against. When provided, the menu renders
   * into a portal at the document body using fixed positioning so it cannot be
   * clipped by ancestors with `overflow: auto/hidden` (e.g. the settings
   * `<main>` scroll container).
   */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

export function ProviderConnectionActionsMenu({
  onAddModels,
  onEdit,
  onDelete,
  onClose,
  anchorRef,
}: ProviderConnectionActionsMenuProps) {
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

  useEffect(() => {
    menuItemsRef.current[0]?.focus();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (anchorElement?.contains(target)) return;
      onClose();
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
  }, [anchorElement, onClose]);

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
      data-testid="provider-connection-actions-menu"
    >
      <MenuItem
        index={0}
        icon={<AddModelsIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_BULK_ADD)}
        onClick={() => handleAction(onAddModels)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="provider-connection-add-models"
      />
      <MenuItem
        index={1}
        icon={<EditIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$EDIT)}
        onClick={() => handleAction(onEdit)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="provider-connection-edit"
      />
      <MenuItem
        index={2}
        icon={<DeleteIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$DELETE)}
        onClick={() => handleAction(onDelete)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="provider-connection-delete"
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
