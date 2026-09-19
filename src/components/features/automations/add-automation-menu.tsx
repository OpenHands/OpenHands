import { useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { ChevronDown, FileUp, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { ContextMenu } from "#/ui/context-menu";
import { KebabMenuItemContent } from "./kebab-menu-item-content";

interface AddAutomationMenuProps {
  onAdd: () => void;
  onImport: () => void;
  isAddDisabled?: boolean;
}

export function AddAutomationMenu({
  onAdd,
  onImport,
  isAddDisabled = false,
}: AddAutomationMenuProps) {
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

      setPortalStyle({
        position: "fixed",
        zIndex: 9999,
        top: rect.bottom + 2,
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

  const closeAnd = (action: () => void) => {
    action();
    setOpen(false);
  };

  const menu =
    open && portalStyle ? (
      <ContextMenu
        ref={menuRef}
        testId="automations-add-automation-menu"
        theme="popover"
        position="none"
        alignment="none"
        spacing="none"
        className="min-w-[10rem]"
      >
        <li>
          <ContextMenuListItem
            testId="automations-add-automation-create"
            isDisabled={isAddDisabled}
            className="group"
            onClick={() => {
              if (isAddDisabled) return;
              closeAnd(onAdd);
            }}
          >
            <KebabMenuItemContent
              icon={<Plus className="size-4" aria-hidden />}
              label={t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
            />
          </ContextMenuListItem>
        </li>
        <li>
          <ContextMenuListItem
            testId="automations-import-automation"
            className="group"
            onClick={() => closeAnd(onImport)}
          >
            <KebabMenuItemContent
              icon={<FileUp className="size-4" aria-hidden />}
              label={t(I18nKey.AUTOMATIONS$IMPORT)}
            />
          </ContextMenuListItem>
        </li>
      </ContextMenu>
    ) : null;

  return (
    <>
      <BrandButton
        ref={triggerRef}
        type="button"
        variant="secondary"
        testId="automations-add-automation"
        className="whitespace-nowrap"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {t(I18nKey.AUTOMATIONS$ADD_AUTOMATION)}
        <ChevronDown className="size-4 shrink-0" aria-hidden />
      </BrandButton>

      {open && portalStyle && typeof document !== "undefined"
        ? ReactDOM.createPortal(
            <div style={portalStyle}>{menu}</div>,
            document.body,
          )
        : null}
    </>
  );
}
