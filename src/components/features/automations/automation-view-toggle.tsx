import { useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { Grid2x2, Rows3 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import { cn } from "#/utils/utils";
import type { AutomationViewMode } from "./automation-view-mode";
import { ViewMenuItemContent } from "./view-menu-item-content";

interface AutomationViewToggleProps {
  view: AutomationViewMode;
  onChange: (view: AutomationViewMode) => void;
  disabled?: boolean;
}

const VIEW_OPTIONS: {
  value: AutomationViewMode;
  icon: typeof Grid2x2;
  labelKey: I18nKey;
  testId: string;
}[] = [
  {
    value: "grid",
    icon: Grid2x2,
    labelKey: I18nKey.AUTOMATIONS$VIEW_GRID,
    testId: "automations-view-toggle-grid",
  },
  {
    value: "list",
    icon: Rows3,
    labelKey: I18nKey.AUTOMATIONS$VIEW_LIST,
    testId: "automations-view-toggle-list",
  },
];

export function AutomationViewToggle({
  view,
  onChange,
  disabled = false,
}: AutomationViewToggleProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const [portalStyle, setPortalStyle] = useState<React.CSSProperties>();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);

  const activeOption =
    VIEW_OPTIONS.find((option) => option.value === view) ?? VIEW_OPTIONS[0]!;
  const ActiveIcon = activeOption.icon;

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
        {VIEW_OPTIONS.map((option) => (
          <li key={option.value}>
            <ContextMenuListItem
              testId={option.testId}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
              className="group"
            >
              <ViewMenuItemContent
                icon={option.icon}
                label={t(option.labelKey)}
                isSelected={view === option.value}
              />
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
        data-testid="automations-view-toggle"
        aria-label={t(I18nKey.AUTOMATIONS$VIEW_MODE)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-disabled={disabled}
        disabled={disabled}
        onClick={() => {
          if (disabled) return;
          setOpen((current) => !current);
        }}
        className={cn(
          "inline-flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-lg border border-[var(--oh-border)] bg-base-secondary text-white transition-colors hover:bg-[var(--oh-interactive-hover)] focus-visible:border-white/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
          "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-base-secondary",
        )}
      >
        <ActiveIcon className="size-4" aria-hidden />
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
