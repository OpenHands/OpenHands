import {
  useEffect,
  useLayoutEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { KebabMenuItemContent } from "#/components/features/automations/kebab-menu-item-content";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { MenuSeparator } from "#/components/features/conversation-panel/menu-separator";
import { ContextMenu } from "#/ui/context-menu";
import type { HomeAutomationMenuEntry } from "./build-home-automation-menu-items";

interface HomeAutomationMenuProps {
  testId: string;
  panelTestId: string;
  ariaLabel: string;
  items: HomeAutomationMenuEntry[];
  triggerClassName?: string;
}

/**
 * Shared home kebab: portal menu, Escape/arrow keys, focus restore to trigger.
 */
export function HomeAutomationMenu({
  testId,
  panelTestId,
  ariaLabel,
  items,
  triggerClassName,
}: HomeAutomationMenuProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [, bumpPosition] = useReducer((i: number) => i + 1, 0);

  const closeMenu = () => {
    setMenuOpen(false);
    // Restore focus so keyboard users land back on the ellipsis.
    queueMicrotask(() => anchorRef.current?.focus());
  };

  const menuRef = useClickOutsideElement<HTMLUListElement>(
    closeMenu,
    anchorRef,
  );

  const actionableIndexes = items
    .map((entry, index) =>
      entry.kind === "item" && !entry.disabled ? index : -1,
    )
    .filter((index) => index >= 0);

  useLayoutEffect(() => {
    if (!menuOpen) return undefined;
    bumpPosition();
    setActiveIndex(actionableIndexes[0] ?? 0);
    let frame = 0;
    const update = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        bumpPosition();
      });
    };
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      if (frame) window.cancelAnimationFrame(frame);
    };
    // actionableIndexes is derived from items; re-run when the menu opens.
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    itemRefs.current[activeIndex]?.focus();
  }, [activeIndex, menuOpen]);

  const floatingStyle = (() => {
    if (!menuOpen || !anchorRef.current) return undefined;
    const rect = anchorRef.current.getBoundingClientRect();
    const gutter = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // The panel's real height is only measurable from the second render
    // pass onward (the open effect forces one via bumpPosition); the first
    // paint falls back to an items-derived estimate (~36px per row, ~9px
    // per separator, plus panel padding).
    const estimatedHeight = items.reduce(
      (total, entry) => total + (entry.kind === "separator" ? 9 : 36),
      10,
    );
    const menuHeight =
      menuRef.current?.getBoundingClientRect().height ?? estimatedHeight;
    const overflowsBelow = rect.bottom + 4 + menuHeight + gutter > vh;
    return {
      position: "fixed" as const,
      // When the menu would clip at the viewport bottom, flip it above the
      // trigger. Bottom-anchoring (rather than an estimated `top`) keeps
      // the panel hugging the trigger whatever its actual height.
      ...(overflowsBelow
        ? { bottom: vh - rect.top + 4 }
        : { top: rect.bottom + 4 }),
      right: Math.max(gutter, vw - rect.right),
      zIndex: 100_000,
    };
  })();

  const portalTarget = typeof document !== "undefined" ? document.body : null;

  const moveActive = (direction: 1 | -1) => {
    if (actionableIndexes.length === 0) return;
    const currentPos = actionableIndexes.indexOf(activeIndex);
    const nextPos =
      (currentPos + direction + actionableIndexes.length) %
      actionableIndexes.length;
    setActiveIndex(actionableIndexes[nextPos] ?? actionableIndexes[0]);
  };

  return (
    <div className="relative shrink-0">
      <EllipsisButton
        ref={anchorRef}
        testId={testId}
        ariaLabel={ariaLabel}
        className={triggerClassName}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setMenuOpen((open) => !open);
        }}
      />

      {menuOpen && floatingStyle && portalTarget
        ? createPortal(
            <ContextMenu
              ref={menuRef}
              testId={panelTestId}
              theme="popover"
              position="none"
              alignment="none"
              spacing="none"
              style={floatingStyle}
              className="min-w-40"
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.preventDefault();
                  event.stopPropagation();
                  closeMenu();
                  return;
                }
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  moveActive(1);
                  return;
                }
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  moveActive(-1);
                }
              }}
            >
              {items.map((entry, index) => {
                if (entry.kind === "separator") {
                  return <MenuSeparator key={entry.key} />;
                }

                return (
                  <li key={entry.key}>
                    <ContextMenuListItem
                      ref={(node) => {
                        itemRefs.current[index] = node;
                      }}
                      testId={entry.testId}
                      isDisabled={entry.disabled}
                      className="group"
                      onClick={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (entry.disabled) return;
                        entry.onClick();
                        closeMenu();
                      }}
                    >
                      <KebabMenuItemContent
                        icon={entry.icon}
                        label={entry.label}
                      />
                    </ContextMenuListItem>
                  </li>
                );
              })}
            </ContextMenu>,
            portalTarget,
          )
        : null}
    </div>
  );
}
