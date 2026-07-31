import React from "react";
import { createPortal } from "react-dom";
import { MessageSquare, Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useConversationSearch } from "#/hooks/query/use-conversation-search";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { HighlightSearchMatch } from "#/components/features/conversation-panel/highlight-search-match";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { useSidebarStore } from "#/stores/sidebar-store";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { matchesPrimaryModifierShortcut } from "#/utils/keyboard-shortcut";
import { cn } from "#/utils/utils";
import {
  COMMAND_MENU_ID,
  CONVERSATION_PANEL_SEARCH_HOTKEY,
} from "#/components/features/conversation-panel/conversation-panel-search-constants";
import {
  COMMAND_MENU_GROUP_LABELS,
  COMMAND_MENU_GROUP_ORDER,
  COMMAND_MENU_GROUP_ORDER_WITH_CONVERSATIONS,
  type CommandMenuGroupId,
  type CommandMenuItemDefinition,
  createCommandMenuItems,
} from "./command-menu-items";

const COMMAND_MENU_SEARCH_INPUT_ID = "command-menu-search";
const COMMAND_MENU_LISTBOX_ID = "command-menu-results";
const COMMAND_MENU_OPTION_ID_PREFIX = "command-menu-option";
const COMMAND_MENU_ARROW_DOWN_KEY = "ArrowDown";
const COMMAND_MENU_ARROW_UP_KEY = "ArrowUp";
const COMMAND_MENU_ENTER_KEY = "Enter";
const COMMAND_MENU_ESCAPE_KEY = "Escape";
const EMPTY_QUERY = "";
const EMPTY_RESULTS_ACTIVE_INDEX = -1;
const CONVERSATION_RESULT_LIMIT = 8;
const CONVERSATION_ROUTE_PREFIX = "/conversations/";

type CommandMenuEntry =
  | { kind: "command"; item: CommandMenuItemDefinition }
  | { kind: "conversation"; conversation: AppConversation };

function getOptionId(entry: CommandMenuEntry) {
  if (entry.kind === "conversation") {
    return `${COMMAND_MENU_OPTION_ID_PREFIX}-conversation-${entry.conversation.id}`;
  }
  return `${COMMAND_MENU_OPTION_ID_PREFIX}-${entry.item.id}`;
}

function getEntryKey(entry: CommandMenuEntry) {
  return entry.kind === "conversation"
    ? `conversation:${entry.conversation.id}`
    : `command:${entry.item.id}`;
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

function getConversationContextLabel(
  conversation: AppConversation,
): string | null {
  if (conversation.selected_repository) {
    const parts = conversation.selected_repository.split("/");
    return parts[parts.length - 1] ?? conversation.selected_repository;
  }

  const workspacePath =
    conversation.selected_workspace ?? conversation.workspace?.working_dir;
  if (!workspacePath) {
    return null;
  }

  const segments = workspacePath.split("/").filter(Boolean);
  return segments[segments.length - 1] ?? workspacePath;
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

  const trimmedQuery = query.trim();
  const conversationSearchEnabled = isOpen && trimmedQuery.length > 0;
  const {
    data: conversationMatches = [],
    isFetching: isFetchingConversations,
    isError: isConversationSearchError,
  } = useConversationSearch(query, conversationSearchEnabled);

  const conversationResults = React.useMemo(
    () => conversationMatches.slice(0, CONVERSATION_RESULT_LIMIT),
    [conversationMatches],
  );
  const hasMoreConversationMatches =
    conversationMatches.length > CONVERSATION_RESULT_LIMIT;

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        matchesPrimaryModifierShortcut(event, CONVERSATION_PANEL_SEARCH_HOTKEY)
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

  const entries = React.useMemo<CommandMenuEntry[]>(() => {
    const conversationEntries: CommandMenuEntry[] = conversationResults.map(
      (conversation) => ({ kind: "conversation", conversation }),
    );
    const commandEntries: CommandMenuEntry[] = filteredItems.map((item) => ({
      kind: "command",
      item,
    }));
    return [...conversationEntries, ...commandEntries];
  }, [conversationResults, filteredItems]);

  const showConversationsSection =
    conversationSearchEnabled &&
    (conversationResults.length > 0 ||
      isFetchingConversations ||
      isConversationSearchError);

  const groupOrder: CommandMenuGroupId[] = showConversationsSection
    ? COMMAND_MENU_GROUP_ORDER_WITH_CONVERSATIONS
    : COMMAND_MENU_GROUP_ORDER;

  React.useEffect(() => {
    setActiveIndex((currentIndex) => {
      if (entries.length === 0) {
        return EMPTY_RESULTS_ACTIVE_INDEX;
      }
      return Math.min(Math.max(currentIndex, 0), entries.length - 1);
    });
  }, [entries.length]);

  React.useEffect(() => {
    const activeEntry = entries[activeIndex];
    if (!activeEntry) {
      return;
    }

    const activeNode = optionRefs.current.get(getEntryKey(activeEntry));
    if (typeof activeNode?.scrollIntoView === "function") {
      activeNode.scrollIntoView({
        block: "nearest",
      });
    }
  }, [activeIndex, entries]);

  const runEntry = React.useCallback(
    (entry: CommandMenuEntry | undefined) => {
      if (!entry) {
        return;
      }

      close();
      if (entry.kind === "conversation") {
        navigate(`${CONVERSATION_ROUTE_PREFIX}${entry.conversation.id}`);
        return;
      }
      if (entry.item.to) {
        navigate(entry.item.to);
        return;
      }
      entry.item.perform?.();
    },
    [close, navigate],
  );

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === COMMAND_MENU_ARROW_DOWN_KEY) {
      event.preventDefault();
      setActiveIndex((index) =>
        entries.length === 0 ? index : (index + 1) % entries.length,
      );
      return;
    }

    if (event.key === COMMAND_MENU_ARROW_UP_KEY) {
      event.preventDefault();
      setActiveIndex((index) =>
        entries.length === 0
          ? index
          : (index - 1 + entries.length) % entries.length,
      );
      return;
    }

    if (event.key === COMMAND_MENU_ENTER_KEY) {
      event.preventDefault();
      runEntry(entries[activeIndex]);
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

  const activeEntry = entries[activeIndex];
  const showEmptyState =
    entries.length === 0 &&
    !isFetchingConversations &&
    !(conversationSearchEnabled && isConversationSearchError);

  return createPortal(
    <div
      id={COMMAND_MENU_ID}
      className="fixed inset-0 z-[70] flex items-start justify-center px-3 pt-[10vh] sm:px-6"
      data-testid={COMMAND_MENU_ID}
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
            data-testid="command-menu-search-input"
            className="h-11 min-w-0 flex-1 bg-transparent text-base text-white outline-none placeholder:text-[var(--oh-text-dim)]"
            placeholder={t(I18nKey.COMMAND_MENU$PLACEHOLDER)}
            aria-label={t(I18nKey.COMMAND_MENU$SEARCH_LABEL)}
            role="combobox"
            aria-expanded="true"
            aria-controls={COMMAND_MENU_LISTBOX_ID}
            aria-activedescendant={
              activeEntry ? getOptionId(activeEntry) : undefined
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
          {showEmptyState ? (
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <div className="flex size-11 items-center justify-center rounded-2xl border border-dashed border-[var(--oh-border)] text-[var(--oh-text-dim)]">
                <Search className="size-5" />
              </div>
              <p className="text-sm font-medium text-white">
                {trimmedQuery.length > 0
                  ? t(I18nKey.CONVERSATION_PANEL$SEARCH_NO_RESULTS)
                  : t(I18nKey.COMMAND_MENU$NO_RESULTS_TITLE)}
              </p>
              <p className="max-w-sm text-xs leading-5 text-[var(--oh-muted)]">
                {t(I18nKey.COMMAND_MENU$NO_RESULTS_DESCRIPTION)}
              </p>
            </div>
          ) : (
            groupOrder.map((groupId) => {
              if (groupId === "conversations") {
                if (!showConversationsSection) {
                  return null;
                }

                return (
                  <section
                    key={groupId}
                    className="py-1"
                    data-testid="command-menu-conversations-section"
                  >
                    <div className="flex items-baseline justify-between gap-2 px-3 pb-1 pt-2">
                      <span className="text-[11px] font-medium text-[var(--oh-text-dim)]">
                        {t(COMMAND_MENU_GROUP_LABELS.conversations)}
                      </span>
                      {hasMoreConversationMatches ? (
                        <span
                          className="text-[10px] text-[var(--oh-text-dim)]"
                          data-testid="command-menu-conversations-truncated-hint"
                        >
                          {t(I18nKey.COMMAND_MENU$SHOWING_FIRST_N, {
                            count: CONVERSATION_RESULT_LIMIT,
                          })}
                        </span>
                      ) : null}
                    </div>
                    {isFetchingConversations &&
                    conversationResults.length === 0 ? (
                      <div
                        className="flex justify-center px-2 py-6"
                        data-testid="command-menu-conversations-loading"
                      >
                        <LoadingSpinner size="small" />
                      </div>
                    ) : null}
                    {isConversationSearchError ? (
                      <p className="px-3 py-4 text-xs text-[var(--oh-muted)]">
                        {t(I18nKey.COMMON$ERROR)}
                      </p>
                    ) : null}
                    <div className="space-y-1">
                      {conversationResults.map((conversation) => {
                        const entry: CommandMenuEntry = {
                          kind: "conversation",
                          conversation,
                        };
                        const entryIndex = entries.findIndex(
                          (candidate) =>
                            candidate.kind === "conversation" &&
                            candidate.conversation.id === conversation.id,
                        );
                        const isActive = entryIndex === activeIndex;
                        const title =
                          conversation.title?.trim() || conversation.id;
                        const contextLabel =
                          getConversationContextLabel(conversation);
                        const timestamp =
                          conversation.updated_at ?? conversation.created_at;
                        const entryKey = getEntryKey(entry);

                        return (
                          <button
                            key={entryKey}
                            ref={(node) => {
                              if (node) {
                                optionRefs.current.set(entryKey, node);
                              } else {
                                optionRefs.current.delete(entryKey);
                              }
                            }}
                            id={getOptionId(entry)}
                            type="button"
                            role="option"
                            aria-selected={isActive}
                            data-testid="command-menu-conversation-result"
                            data-conversation-id={conversation.id}
                            onMouseEnter={() => setActiveIndex(entryIndex)}
                            onClick={() => runEntry(entry)}
                            className={cn(
                              "group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors duration-150",
                              isActive
                                ? "bg-white/[0.09] text-white shadow-[0_0_0_1px_rgba(255,255,255,0.08)_inset]"
                                : "text-[var(--oh-muted)] hover:bg-white/[0.05] hover:text-white",
                            )}
                          >
                            <span
                              className={cn(
                                "flex size-9 shrink-0 items-center justify-center rounded-lg border transition-colors duration-150",
                                isActive
                                  ? "border-white/25 bg-white/10 text-white"
                                  : "border-[var(--oh-border)] bg-black/15 text-[var(--oh-text-dim)] group-hover:text-white",
                              )}
                              aria-hidden="true"
                            >
                              <MessageSquare size={18} />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-sm font-medium text-current">
                                <HighlightSearchMatch
                                  text={title}
                                  query={trimmedQuery}
                                />
                              </span>
                              {contextLabel ? (
                                <span className="mt-0.5 block truncate text-xs text-[var(--oh-text-dim)]">
                                  {contextLabel}
                                </span>
                              ) : null}
                            </span>
                            {timestamp ? (
                              <time
                                dateTime={timestamp}
                                className="hidden shrink-0 text-[10px] text-[var(--oh-text-dim)] sm:inline-flex"
                              >
                                {formatTimeDelta(timestamp)}
                              </time>
                            ) : null}
                            <span className="hidden shrink-0 rounded-md border border-[var(--oh-border)] px-2 py-1 text-[10px] font-medium text-[var(--oh-text-dim)] sm:inline-flex">
                              {t(I18nKey.COMMAND_MENU$OPEN_CONVERSATION_HINT)}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </section>
                );
              }

              const groupItems = filteredItems.filter(
                (item) => item.group === groupId,
              );

              if (groupItems.length === 0) {
                return null;
              }

              return (
                <section key={groupId} className="py-1">
                  <div className="px-3 pb-1 pt-2 text-[11px] font-medium text-[var(--oh-text-dim)]">
                    {t(COMMAND_MENU_GROUP_LABELS[groupId])}
                  </div>
                  <div className="space-y-1">
                    {groupItems.map((item) => {
                      const entry: CommandMenuEntry = {
                        kind: "command",
                        item,
                      };
                      const entryIndex = entries.findIndex(
                        (candidate) =>
                          candidate.kind === "command" &&
                          candidate.item.id === item.id,
                      );
                      const isActive = entryIndex === activeIndex;
                      const to = item.to;
                      const entryKey = getEntryKey(entry);

                      const assignRef = (node: HTMLElement | null) => {
                        if (node) {
                          optionRefs.current.set(entryKey, node);
                        } else {
                          optionRefs.current.delete(entryKey);
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
                                ? "border-white/25 bg-white/10 text-white"
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
                          <span className="hidden shrink-0 rounded-md border border-[var(--oh-border)] px-2 py-1 text-[10px] font-medium text-[var(--oh-text-dim)] sm:inline-flex">
                            {to
                              ? t(I18nKey.COMMAND_MENU$GO_HINT)
                              : t(I18nKey.COMMAND_MENU$RUN_HINT)}
                          </span>
                        </>
                      );

                      if (to) {
                        return (
                          <a
                            key={entryKey}
                            ref={assignRef}
                            id={getOptionId(entry)}
                            href={to}
                            role="option"
                            aria-selected={isActive}
                            onMouseEnter={() => setActiveIndex(entryIndex)}
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
                              runEntry(entry);
                            }}
                            className={optionClassName}
                          >
                            {content}
                          </a>
                        );
                      }

                      return (
                        <button
                          key={entryKey}
                          ref={assignRef}
                          id={getOptionId(entry)}
                          type="button"
                          role="option"
                          aria-selected={isActive}
                          onMouseEnter={() => setActiveIndex(entryIndex)}
                          onClick={() => runEntry(entry)}
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
