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
import CheckCircleIcon from "#/icons/u-check-circle.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import { MenuItem } from "#/components/features/settings/llm-profiles/profile-actions-menu-item";

interface AgentProfileActionsMenuProps {
  onEdit: () => void;
  onSetActive: () => void;
  onDelete: () => void;
  isActive: boolean;
  isActivating: boolean;
  onClose: () => void;
  /**
   * Element the menu anchors against. When provided, the menu renders into a
   * body portal with fixed positioning so it isn't clipped by scroll
   * containers (matches the LLM-profiles menu behavior).
   */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

export function AgentProfileActionsMenu({
  onEdit,
  onSetActive,
  onDelete,
  isActive,
  isActivating,
  onClose,
  anchorRef,
}: AgentProfileActionsMenuProps) {
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
      if (menuRef.current && !menuRef.current.contains(target)) {
        onClose();
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
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
      data-testid="agent-profile-actions-menu"
    >
      <MenuItem
        index={0}
        icon={<EditIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$PROFILE_EDIT)}
        onClick={() => handleAction(onEdit)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="agent-profile-edit"
      />
      <MenuItem
        index={1}
        icon={<CheckCircleIcon width={16} height={16} />}
        label={t(I18nKey.SETTINGS$PROFILE_SET_ACTIVE)}
        onClick={() => handleAction(onSetActive)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        disabled={setActiveDisabled}
        testId="agent-profile-set-active"
      />
      <MenuItem
        index={2}
        icon={<DeleteIcon width={16} height={16} />}
        label={t(I18nKey.BUTTON$DELETE)}
        onClick={() => handleAction(onDelete)}
        onKeyDown={handleKeyDown}
        menuItemsRef={menuItemsRef}
        testId="agent-profile-delete"
      />
    </div>
  );

  if (isPortaled) {
    if (typeof document === "undefined" || !portalStyle) {
      return null;
    }
    return ReactDOM.createPortal(
      <div style={portalStyle}>{menu}</div>,
      document.body,
    );
  }

  return menu;
}
