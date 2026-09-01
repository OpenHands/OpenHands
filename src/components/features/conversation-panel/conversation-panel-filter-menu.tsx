import React from "react";
import { useTranslation } from "react-i18next";
import {
  CircleDot,
  Clock3,
  Folder,
  ListFilter,
  SlidersHorizontal,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  dropdownInstantColorClassName,
  dropdownMenuListClassName,
  dropdownMenuViewportScrollClassName,
} from "#/utils/dropdown-classes";
import {
  ConversationPanelAdvancedOptions,
  type ConversationPanelDisplaySettingsProps,
} from "./conversation-panel-advanced-options";
import { MenuRow } from "./menu-row";

export interface ConversationPanelFilterMenuProps extends ConversationPanelDisplaySettingsProps {
  filterMenuOpen: boolean;
  setFilterMenuOpen: (open: boolean) => void;
}

export function ConversationPanelFilterMenu({
  filterMenuOpen,
  setFilterMenuOpen,
  backendKind,
  organizeMode,
  setOrganizeMode,
  conversationSort,
  setConversationSort,
  threadScope,
  setThreadScope,
  automationFilterMode,
  setAutomationFilterMode,
  selectedAutomationNames,
  onToggleAutomationName,
  automationNameFacets,
  showOlderConversations,
  showArchivedConversations,
  toggleShowArchivedConversations,
  toggleShowOlderConversations,
  showRepoBranchMetadata,
  toggleShowRepoBranchMetadata,
  showLlmProfiles,
  toggleShowLlmProfiles,
  showTagsMetadata,
  toggleShowTagsMetadata,
  showHoverMetadata,
  toggleShowHoverMetadata,
  totalConversationsCount,
  onRequestDeleteAll,
}: ConversationPanelFilterMenuProps) {
  const { t } = useTranslation("openhands");
  const [advancedOpen, setAdvancedOpen] = React.useState(false);

  const groupedLabel =
    backendKind === "local"
      ? t(I18nKey.CONVERSATION_PANEL$BY_WORKSPACE)
      : t(I18nKey.CONVERSATION_PANEL$BY_REPOSITORY);

  const menuRef = React.useRef<HTMLDivElement>(null);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const menuContentRef = React.useRef<HTMLDivElement>(null);

  const wasOpenRef = React.useRef(filterMenuOpen);
  React.useEffect(() => {
    if (!filterMenuOpen) {
      setAdvancedOpen(false);
    }
    if (filterMenuOpen) {
      const firstItem =
        menuContentRef.current?.querySelector<HTMLButtonElement>(
          '[role="menuitem"], [role="menuitemradio"]',
        );
      firstItem?.focus();
    } else if (wasOpenRef.current) {
      triggerRef.current?.focus();
    }
    wasOpenRef.current = filterMenuOpen;
  }, [filterMenuOpen, advancedOpen]);

  React.useEffect(() => {
    if (!filterMenuOpen) return undefined;

    const handlePointerDownOutside = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && menuRef.current?.contains(target)) return;
      setFilterMenuOpen(false);
    };

    document.addEventListener("pointerdown", handlePointerDownOutside);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDownOutside);
    };
  }, [filterMenuOpen, setFilterMenuOpen]);

  const handleMenuKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setFilterMenuOpen(false);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const container = menuContentRef.current;
    if (!container) return;
    const items = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[role="menuitem"], [role="menuitemradio"]',
      ),
    ).filter((el) => !el.disabled);
    if (items.length === 0) return;
    const currentIdx = items.indexOf(
      document.activeElement as HTMLButtonElement,
    );
    const delta = event.key === "ArrowDown" ? 1 : -1;
    const start = currentIdx === -1 ? 0 : currentIdx;
    const nextIdx = (start + delta + items.length) % items.length;
    event.preventDefault();
    items[nextIdx]?.focus();
  };

  const handleSelectOrganize = (mode: typeof organizeMode) => {
    setOrganizeMode(mode);
    setFilterMenuOpen(false);
  };

  const handleSelectShowActive = () => {
    setThreadScope(threadScope === "relevant" ? "all" : "relevant");
    setFilterMenuOpen(false);
  };

  const handleClose = () => {
    setAdvancedOpen(false);
    setFilterMenuOpen(false);
  };

  const displaySettings: ConversationPanelDisplaySettingsProps = {
    backendKind,
    organizeMode,
    setOrganizeMode,
    conversationSort,
    setConversationSort,
    threadScope,
    setThreadScope,
    automationFilterMode,
    setAutomationFilterMode,
    selectedAutomationNames,
    onToggleAutomationName,
    automationNameFacets,
    showOlderConversations,
    showArchivedConversations,
    toggleShowArchivedConversations,
    toggleShowOlderConversations,
    showRepoBranchMetadata,
    toggleShowRepoBranchMetadata,
    showLlmProfiles,
    toggleShowLlmProfiles,
    showTagsMetadata,
    toggleShowTagsMetadata,
    showHoverMetadata,
    toggleShowHoverMetadata,
    totalConversationsCount,
    onRequestDeleteAll,
  };

  return (
    <div ref={menuRef} className="relative shrink-0 pr-0.5">
      <button
        ref={triggerRef}
        type="button"
        data-testid="older-conversations-filter-toggle"
        aria-label={t(I18nKey.CONVERSATION_PANEL$FILTER_LABEL)}
        aria-haspopup="menu"
        aria-expanded={filterMenuOpen}
        onClick={() => setFilterMenuOpen(!filterMenuOpen)}
        className={cn(
          "relative inline-flex h-7 w-7 items-center justify-center rounded-md text-[var(--oh-muted)] hover:text-white hover:bg-[var(--oh-surface-raised)]",
          dropdownInstantColorClassName,
        )}
      >
        <ListFilter
          className="lucide lucide-list-filter shrink-0"
          width={14}
          height={14}
          strokeWidth={2}
          aria-hidden
        />
        {automationFilterMode !== "all" ? (
          <span
            aria-hidden
            data-testid="automation-filter-active-indicator"
            className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-[var(--oh-accent)]"
          />
        ) : null}
      </button>

      {filterMenuOpen ? (
        <div
          ref={menuContentRef}
          role="menu"
          aria-orientation="vertical"
          aria-label={
            advancedOpen
              ? t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)
              : t(I18nKey.CONVERSATION_PANEL$FILTER_LABEL)
          }
          tabIndex={-1}
          data-testid={
            advancedOpen
              ? "conversation-advanced-options"
              : "older-conversations-filter-menu"
          }
          onKeyDown={handleMenuKeyDown}
          className={cn(
            "absolute right-0 top-full z-50 mt-0 w-64 rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 text-[var(--oh-foreground)] shadow-lg",
            dropdownMenuListClassName,
            advancedOpen ? dropdownMenuViewportScrollClassName : null,
          )}
        >
          {advancedOpen ? (
            <ConversationPanelAdvancedOptions
              {...displaySettings}
              onClose={handleClose}
            />
          ) : (
            <>
              <MenuRow
                icon={Folder}
                label={groupedLabel}
                selected={organizeMode === "grouped"}
                onClick={() => handleSelectOrganize("grouped")}
              />
              <MenuRow
                icon={Clock3}
                label={t(I18nKey.CONVERSATION_PANEL$CHRONOLOGICAL)}
                selected={organizeMode === "chronological"}
                onClick={() => handleSelectOrganize("chronological")}
              />
              <MenuRow
                icon={CircleDot}
                label={t(I18nKey.CONVERSATION_PANEL$SHOW_ACTIVE)}
                selected={threadScope === "relevant"}
                testId="conversation-layout-show-active"
                onClick={handleSelectShowActive}
              />
              <MenuRow
                icon={SlidersHorizontal}
                label={t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)}
                testId="conversation-layout-advanced"
                onClick={() => setAdvancedOpen(true)}
              />
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
