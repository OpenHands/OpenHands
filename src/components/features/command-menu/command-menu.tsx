import React from "react";
import { createPortal } from "react-dom";
import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { cn } from "#/utils/utils";
import {
  COMMAND_MENU_GROUP_LABELS,
  COMMAND_MENU_GROUP_ORDER,
  type CommandMenuItemDefinition,
  createCommandMenuItems,
} from "./command-menu-items";

const COMMAND_MENU_SEARCH_INPUT_ID = "command-menu-search";
const COMMAND_MENU_LISTBOX_ID = "command-menu-results";
const COMMAND_MENU_OPTION_ID_PREFIX = "command-menu-option";
const COMMAND_MENU_TEST_ID = "command-menu";
const COMMAND_MENU_SHORTCUT_KEY = "k";
const COMMAND_MENU_ARROW_DOWN_KEY = "ArrowDown";
const COMMAND_MENU_ARROW_UP_KEY = "ArrowUp";
const COMMAND_MENU_ENTER_KEY = "Enter";
const COMMAND_MENU_ESCAPE_KEY = "Escape";
const EMPTY_QUERY = "";
const EMPTY_RESULTS_ACTIVE_INDEX = -1;

function getOptionId(item: CommandMenuItemDefinition) {
  return `${COMMAND_MENU_OPTION_ID_PREFIX}-${item.id}`;
}

function matchesQuery({
  item,
  query,
  translate,
}: {
  item: CommandMenuItemDefinition;
  query: string;
  translate: (key: I18nKey) => string;
}) {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);

  if (terms.length === 0) {
    return true;
  }

  const searchableText = [
    translate(item.titleKey),
    translate(item.descriptionKey),
    translate(item.keywordsKey),
  ]
    .join(" ")
    .toLocaleLowerCase();

  return terms.every((term) => searchableText.includes(term));
}

export function CommandMenu() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const isOpen = useCommandMenuStore((state) => state.isOpen);
  const open = useCommandMenuStore((state) => state.open);
  const close = useCommandMenuStore((state) => state.close);
  const [query, setQuery] = React.useState(EMPTY_QUERY);
  const [activeIndex, setActiveIndex] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const optionRefs = React.useRef(new Map<string, HTMLElement>());

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        (event.metaKey || event.ctrlKey) &&
        event.key.toLocaleLowerCase() === COMMAND_MENU_SHORTCUT_KEY
      ) {
        event.preventDefault();
        open();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  React.useEffect(() => {
    if (!isOpen) {
      setQuery(EMPTY_QUERY);
      setActiveIndex(0);
      optionRefs.current.clear();
      return undefined;
    }

    const frame = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [isOpen]);

  const items = React.useMemo(
    () =>
      createCommandMenuItems({
        toggleSidebar: () => useSidebarStore.getState().toggleCollapsed(),
      }),
    [],
  );

  const filteredItems = React.useMemo(
    () =>
      items.filter((item) =>
        matchesQuery({ item, query, translate: (key) => t(key) }),
      ),
    [items, query, t],
  );

  React.useEffect(() => {
    setActiveIndex((currentIndex) => {
      if (filteredItems.length === 0) {
        return EMPTY_RESULTS_ACTIVE_INDEX;
      }
      return Math.min(Math.max(currentIndex, 0), filteredItems.length - 1);
    });
  }, [filteredItems.length]);

  React.useEffect(() => {
    const activeItem = filteredItems[activeIndex];
    if (!activeItem) {
      return;
    }

    const activeNode = optionRefs.current.get(activeItem.id);
    if (typeof activeNode?.scrollIntoView === "function") {
      activeNode.scrollIntoView({
        block: "nearest",
      });
    }
  }, [activeIndex, filteredItems]);

  const runItem = React.useCallback(
    (item: CommandMenuItemDefinition | undefined) => {
      if (!item) {
        return;
      }

      close();
      if (item.to) {
        navigate(item.to);
        return;
      }
      item.perform?.();
    },
    [close, navigate],
  );

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === COMMAND_MENU_ARROW_DOWN_KEY) {
      event.preventDefault();
      setActiveIndex((index) =>
        filteredItems.length === 0 ? index : (index + 1) % filteredItems.length,
      );
      return;
    }

    if (event.key === COMMAND_MENU_ARROW_UP_KEY) {
      event.preventDefault();
      setActiveIndex((index) =>
        filteredItems.length === 0
          ? index
          : (index - 1 + filteredItems.length) % filteredItems.length,
      );
      return;
    }

    if (event.key === COMMAND_MENU_ENTER_KEY) {
      event.preventDefault();
      runItem(filteredItems[activeIndex]);
      return;
    }

    if (event.key === COMMAND_MENU_ESCAPE_KEY) {
      event.preventDefault();
      close();
    }
  };

  if (!isOpen || typeof document === "undefined") {
    return null;
  }

  const activeItem = filteredItems[activeIndex];

  return createPortal(
    <div
      className="fixed inset-0 z-[70] flex items-start justify-center px-3 pt-[10vh] sm:px-6"
      data-testid={COMMAND_MENU_TEST_ID}
      role="dialog"
      aria-modal="true"
      aria-label={t(I18nKey.COMMAND_MENU$ARIA_LABEL)}
    >
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-black/65 backdrop-blur-[2px]"
        aria-label={t(I18nKey.COMMAND_MENU$CLOSE_LABEL)}
        onClick={close}
      />
      <div
        className={cn(
          "relative flex max-h-[min(720px,78vh)] w-full max-w-2xl flex-col overflow-hidden rounded-2xl",
          "border border-[var(--oh-border)] bg-[var(--oh-surface)]",
          "shadow-[0_24px_90px_rgba(0,0,0,0.52),0_0_0_1px_rgba(255,255,255,0.03)_inset]",
        )}
      >
        <div className="relative flex items-center gap-3 border-b border-[var(--oh-border)] px-4 py-3">
          <Search className="size-5 shrink-0 text-[var(--oh-text-dim)]" />
          <input
            ref={inputRef}
            id={COMMAND_MENU_SEARCH_INPUT_ID}
            className="h-11 min-w-0 flex-1 bg-transparent text-base text-white outline-none placeholder:text-[var(--oh-text-dim)]"
            placeholder={t(I18nKey.COMMAND_MENU$PLACEHOLDER)}
            aria-label={t(I18nKey.COMMAND_MENU$SEARCH_LABEL)}
            role="combobox"
            aria-expanded="true"
            aria-controls={COMMAND_MENU_LISTBOX_ID}
            aria-activedescendant={
              activeItem ? getOptionId(activeItem) : undefined
            }
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleInputKeyDown}
          />
          {query ? (
            <button
              type="button"
              className="inline-flex size-8 items-center justify-center rounded-lg text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-white"
              aria-label={t(I18nKey.COMMAND_MENU$CLEAR_SEARCH_LABEL)}
              onClick={() => {
                setQuery(EMPTY_QUERY);
                inputRef.current?.focus();
              }}
            >
              <X className="size-4" />
            </button>
          ) : null}
          <kbd className="hidden rounded-md border border-[var(--oh-border)] bg-black/25 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--oh-text-dim)] sm:inline-flex">
            {t(I18nKey.COMMAND_MENU$SHORTCUT)}
          </kbd>
        </div>

        <div
          id={COMMAND_MENU_LISTBOX_ID}
          role="listbox"
          className="relative min-h-0 flex-1 overflow-y-auto px-2 py-2 custom-scrollbar"
        >
          {filteredItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <div className="flex size-11 items-center justify-center rounded-2xl border border-dashed border-[var(--oh-border)] text-[var(--oh-text-dim)]">
                <Search className="size-5" />
              </div>
              <p className="text-sm font-medium text-white">
                {t(I18nKey.COMMAND_MENU$NO_RESULTS_TITLE)}
              </p>
              <p className="max-w-sm text-xs leading-5 text-[var(--oh-muted)]">
                {t(I18nKey.COMMAND_MENU$NO_RESULTS_DESCRIPTION)}
              </p>
            </div>
          ) : (
            COMMAND_MENU_GROUP_ORDER.map((groupId) => {
              const groupItems = filteredItems.filter(
                (item) => item.group === groupId,
              );

              if (groupItems.length === 0) {
                return null;
              }

              return (
                <section key={groupId} className="py-1">
                  <div className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--oh-text-dim)]">
                    {t(COMMAND_MENU_GROUP_LABELS[groupId])}
                  </div>
                  <div className="space-y-1">
                    {groupItems.map((item) => {
                      const itemIndex = filteredItems.indexOf(item);
                      const isActive = itemIndex === activeIndex;
                      const to = item.to;

                      const assignRef = (node: HTMLElement | null) => {
                        if (node) {
                          optionRefs.current.set(item.id, node);
                        } else {
                          optionRefs.current.delete(item.id);
                        }
                      };

                      const optionClassName = cn(
                        "group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors duration-150",
                        isActive
                          ? "bg-white/[0.09] text-white shadow-[0_0_0_1px_rgba(255,255,255,0.08)_inset]"
                          : "text-[var(--oh-muted)] hover:bg-white/[0.05] hover:text-white",
                      );

                      const content = (
                        <>
                          <span
                            className={cn(
                              "flex size-9 shrink-0 items-center justify-center rounded-lg border transition-colors duration-150",
                              isActive
                                ? "border-[var(--oh-accent)] bg-[var(--oh-accent)]/15 text-white"
                                : "border-[var(--oh-border)] bg-black/15 text-[var(--oh-text-dim)] group-hover:text-white",
                            )}
                            aria-hidden="true"
                          >
                            {item.icon}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-current">
                              {t(item.titleKey)}
                            </span>
                            <span className="mt-0.5 block truncate text-xs text-[var(--oh-text-dim)]">
                              {t(item.descriptionKey)}
                            </span>
                          </span>
                          <span className="hidden shrink-0 rounded-md border border-[var(--oh-border)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--oh-text-dim)] sm:inline-flex">
                            {to
                              ? t(I18nKey.COMMAND_MENU$GO_HINT)
                              : t(I18nKey.COMMAND_MENU$RUN_HINT)}
                          </span>
                        </>
                      );

                      if (to) {
                        return (
                          <a
                            key={item.id}
                            ref={assignRef}
                            id={getOptionId(item)}
                            href={to}
                            role="option"
                            aria-selected={isActive}
                            onMouseEnter={() => setActiveIndex(itemIndex)}
                            onClick={(event) => {
                              if (
                                event.metaKey ||
                                event.ctrlKey ||
                                event.shiftKey ||
                                event.altKey
                              ) {
                                return;
                              }
                              event.preventDefault();
                              runItem(item);
                            }}
                            className={optionClassName}
                          >
                            {content}
                          </a>
                        );
                      }

                      return (
                        <button
                          key={item.id}
                          ref={assignRef}
                          id={getOptionId(item)}
                          type="button"
                          role="option"
                          aria-selected={isActive}
                          onMouseEnter={() => setActiveIndex(itemIndex)}
                          onClick={() => runItem(item)}
                          className={optionClassName}
                        >
                          {content}
                        </button>
                      );
                    })}
                  </div>
                </section>
              );
            })
          )}
        </div>

        <div className="border-t border-[var(--oh-border)] px-4 py-2.5 text-[11px] text-[var(--oh-text-dim)]">
          {t(I18nKey.COMMAND_MENU$FOOTER_HINT)}
        </div>
      </div>
    </div>,
    document.body,
  );
}
