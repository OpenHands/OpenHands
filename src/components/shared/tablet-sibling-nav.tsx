import React from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useNavigation } from "#/context/navigation-context";
import { cn } from "#/utils/utils";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { I18nKey } from "#/i18n/declaration";

export interface TabletSiblingNavItem {
  to?: string;
  href?: string;
  label: string;
  icon?: React.ReactNode;
  isExternal?: boolean;
  target?: string;
  rel?: string;
}

export interface TabletSiblingNavProps {
  currentSectionLabel: string;
  currentSectionIcon?: React.ReactNode;
  currentPath: string;
  items: TabletSiblingNavItem[];
  ariaLabel?: string;
  className?: string;
}

export function TabletSiblingNav({
  currentSectionLabel,
  currentSectionIcon,
  currentPath,
  items,
  ariaLabel,
  className,
}: TabletSiblingNavProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [isOpen, setIsOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);

  const siblingItems = React.useMemo(
    () =>
      items.filter((item) =>
        item.to ? item.to !== currentPath : item.href !== currentPath,
      ),
    [items, currentPath],
  );

  const closeMenu = React.useCallback(() => {
    setIsOpen(false);
  }, []);

  useCloseOnEscape(isOpen, closeMenu, triggerRef);

  const containerRef = useClickOutsideElement<HTMLDivElement>(
    closeMenu,
    triggerRef,
  );

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (!isOpen) {
      if (
        event.key === "ArrowDown" ||
        event.key === "Enter" ||
        event.key === " "
      ) {
        event.preventDefault();
        setIsOpen(true);
        // Focus first item on next tick after opening
        setTimeout(() => {
          const firstItem =
            menuRef.current?.querySelector<HTMLElement>('[role="menuitem"]');
          firstItem?.focus();
        }, 0);
      }
      return;
    }

    const menuItems = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [],
    );
    const currentIndex = menuItems.findIndex(
      (el) => el === document.activeElement,
    );

    if (event.key === "ArrowDown") {
      event.preventDefault();
      const nextIndex =
        currentIndex === -1 || currentIndex === menuItems.length - 1
          ? 0
          : currentIndex + 1;
      menuItems[nextIndex]?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      const prevIndex =
        currentIndex <= 0 ? menuItems.length - 1 : currentIndex - 1;
      menuItems[prevIndex]?.focus();
    } else if (event.key === "Home") {
      event.preventDefault();
      menuItems[0]?.focus();
    } else if (event.key === "End") {
      event.preventDefault();
      menuItems[menuItems.length - 1]?.focus();
    } else if (event.key === "Tab") {
      closeMenu();
    }
  };

  if (siblingItems.length === 0) {
    return null;
  }

  const effectiveAriaLabel =
    ariaLabel ??
    t(I18nKey.NAV$SECTION_NAV_LABEL, { section: currentSectionLabel });

  return (
    <div
      ref={containerRef}
      data-testid="tablet-sibling-nav"
      className={cn(
        "hidden md:block lg:hidden w-full relative z-30 mb-4",
        className,
      )}
      onKeyDown={handleKeyDown}
    >
      <div className="relative inline-block text-left w-full sm:w-auto">
        <button
          ref={triggerRef}
          type="button"
          data-testid="tablet-sibling-nav-trigger"
          aria-haspopup="menu"
          aria-expanded={isOpen}
          aria-label={effectiveAriaLabel}
          onClick={() => setIsOpen((prev) => !prev)}
          className={cn(
            "inline-flex items-center justify-between gap-2.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-medium text-contrast hover:bg-surface-raised cursor-pointer transition-colors",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-border",
          )}
        >
          <span className="flex items-center gap-2 truncate">
            {currentSectionIcon ? (
              <span className="size-4 shrink-0 flex items-center justify-center text-muted">
                {currentSectionIcon}
              </span>
            ) : null}
            <span className="truncate">{currentSectionLabel}</span>
          </span>
          <ChevronDown
            className={cn(
              "size-4 shrink-0 text-muted transition-transform duration-200",
              isOpen && "rotate-180",
            )}
            aria-hidden
          />
        </button>

        {isOpen && (
          <div
            ref={menuRef}
            role="menu"
            data-testid="tablet-sibling-nav-menu"
            aria-label={currentSectionLabel}
            className="absolute left-0 mt-1.5 min-w-56 max-w-xs w-max rounded-lg border border-border bg-surface p-1 shadow-lg focus:outline-none"
          >
            {siblingItems.map((item) => {
              const key = item.href ?? item.to ?? item.label;
              if (item.isExternal && item.href) {
                return (
                  <a
                    key={key}
                    role="menuitem"
                    data-testid={`tablet-sibling-nav-item-${item.href}`}
                    href={item.href}
                    target={item.target ?? "_blank"}
                    rel={item.rel ?? "noopener noreferrer"}
                    onClick={closeMenu}
                    className="flex items-center justify-between gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-contrast hover:bg-surface-raised cursor-pointer focus:bg-surface-raised focus:outline-none"
                  >
                    <span className="flex items-center gap-2 truncate">
                      {item.icon ? (
                        <span className="size-4 shrink-0 flex items-center justify-center text-muted">
                          {item.icon}
                        </span>
                      ) : null}
                      <span className="truncate">{item.label}</span>
                    </span>
                    <ExternalLink
                      className="size-3.5 shrink-0 text-muted"
                      aria-hidden
                    />
                  </a>
                );
              }

              return (
                <button
                  key={key}
                  type="button"
                  role="menuitem"
                  data-testid={`tablet-sibling-nav-item-${item.to}`}
                  onClick={() => {
                    closeMenu();
                    if (item.to) {
                      navigate(item.to);
                    }
                  }}
                  className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-contrast hover:bg-surface-raised cursor-pointer focus:bg-surface-raised focus:outline-none"
                >
                  {item.icon ? (
                    <span className="size-4 shrink-0 flex items-center justify-center text-muted">
                      {item.icon}
                    </span>
                  ) : null}
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
