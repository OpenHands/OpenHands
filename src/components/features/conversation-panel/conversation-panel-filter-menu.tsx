import React from "react";
import { useTranslation } from "react-i18next";
import {
  Archive,
  Bot,
  CalendarArrowDown,
  Clock3,
  ClockArrowDown,
  Eye,
  EyeOff,
  Folder,
  GitBranch,
  ListFilter,
  MessageCircle,
  MousePointerClick,
  Star,
  Tag,
  Trash2,
  Workflow,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { BackendKind } from "#/api/backend-registry/types";
import { cn } from "#/utils/utils";
import {
  dropdownInstantColorClassName,
  dropdownMenuListClassName,
  dropdownMenuViewportScrollClassName,
} from "#/utils/dropdown-classes";
import {
  UNNAMED_AUTOMATION_FACET,
  type AutomationFilterMode,
  type ConversationSortField,
  type OrganizeMode,
  type ThreadScope,
} from "./conversation-panel-list-helpers";
import { MenuHeading } from "./menu-heading";
import { MenuSeparator } from "./menu-separator";
import { MenuRow } from "./menu-row";

const capitalizeLabel = (label: string) =>
  label.length > 0 ? label.charAt(0).toUpperCase() + label.slice(1) : label;

export interface ConversationPanelFilterMenuProps {
  filterMenuOpen: boolean;
  setFilterMenuOpen: (open: boolean) => void;
  menuRef: React.RefObject<HTMLDivElement | null>;
  backendKind: BackendKind;
  organizeMode: OrganizeMode;
  setOrganizeMode: (mode: OrganizeMode) => void;
  conversationSort: ConversationSortField;
  setConversationSort: (sort: ConversationSortField) => void;
  threadScope: ThreadScope;
  setThreadScope: (scope: ThreadScope) => void;
  automationFilterMode: AutomationFilterMode;
  setAutomationFilterMode: (mode: AutomationFilterMode) => void;
  selectedAutomationNames: string[];
  onToggleAutomationName: (name: string) => void;
  automationNameFacets: string[];
  selectedTagFacets: string[];
  onToggleTagFacet: (facet: string) => void;
  tagFacets: string[];
  showOlderConversations: boolean;
  showArchivedConversations: boolean;
  toggleShowArchivedConversations: () => void;
  toggleShowOlderConversations: () => void;
  showRepoBranchMetadata: boolean;
  toggleShowRepoBranchMetadata: () => void;
  showLlmProfiles: boolean;
  toggleShowLlmProfiles: () => void;
  showTagsMetadata: boolean;
  toggleShowTagsMetadata: () => void;
  showHoverMetadata: boolean;
  toggleShowHoverMetadata: () => void;
  totalConversationsCount: number;
  onRequestDeleteAll: () => void;
}

export function ConversationPanelFilterMenu({
  filterMenuOpen,
  setFilterMenuOpen,
  menuRef,
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
  selectedTagFacets,
  onToggleTagFacet,
  tagFacets,
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

  const groupedLabel =
    backendKind === "local"
      ? t(I18nKey.CONVERSATION_PANEL$BY_WORKSPACE)
      : t(I18nKey.CONVERSATION_PANEL$BY_REPOSITORY);

  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const menuContentRef = React.useRef<HTMLDivElement>(null);

  // When the menu opens, move keyboard focus into it so screen-reader /
  // keyboard-only users can interact with the options immediately. When
  // it closes, return focus to the trigger so Tab order picks up where
  // the user left off.
  const wasOpenRef = React.useRef(filterMenuOpen);
  React.useEffect(() => {
    if (filterMenuOpen) {
      const firstItem =
        menuContentRef.current?.querySelector<HTMLButtonElement>(
          '[role="menuitem"], [role="menuitemradio"]',
        );
      firstItem?.focus();
    } else if (wasOpenRef.current) {
      // Only return focus on a real open→close transition (not the
      // mount-with-open=false case).
      triggerRef.current?.focus();
    }
    wasOpenRef.current = filterMenuOpen;
  }, [filterMenuOpen]);

  // Roving Arrow Up/Down + Escape across the menu items. Tab still works
  // natively; Escape closes the menu and returns focus to the trigger
  // (via the effect above).
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
        {automationFilterMode === "all" && selectedTagFacets.length > 0 ? (
          <span
            aria-hidden
            data-testid="tag-filter-active-indicator"
            className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-[var(--oh-accent)]"
          />
        ) : null}
      </button>

      {filterMenuOpen ? (
        <div
          ref={menuContentRef}
          role="menu"
          aria-orientation="vertical"
          aria-label={t(I18nKey.CONVERSATION_PANEL$FILTER_LABEL)}
          // `role="menu"` is an interactive ARIA role, so the container
          // must be focusable to satisfy jsx-a11y. `-1` keeps it out of
          // the natural Tab order (the menu items themselves are
          // `<button>`s and tabbable on their own) but still allows the
          // open-effect to focus it / its children programmatically.
          tabIndex={-1}
          data-testid="older-conversations-filter-menu"
          onKeyDown={handleMenuKeyDown}
          className={cn(
            "absolute right-0 top-full z-50 mt-0 w-64 rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 text-[var(--oh-foreground)] shadow-lg",
            dropdownMenuListClassName,
            dropdownMenuViewportScrollClassName,
          )}
        >
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$ORGANIZE)}</MenuHeading>
          <MenuRow
            icon={Folder}
            label={groupedLabel}
            selected={organizeMode === "grouped"}
            onClick={() => {
              setOrganizeMode("grouped");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Clock3}
            label={t(I18nKey.CONVERSATION_PANEL$CHRONOLOGICAL)}
            selected={organizeMode === "chronological"}
            onClick={() => {
              setOrganizeMode("chronological");
              setFilterMenuOpen(false);
            }}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SORT_BY)}</MenuHeading>
          <MenuRow
            icon={CalendarArrowDown}
            label={t(I18nKey.CONVERSATION_PANEL$SORT_CREATED)}
            selected={conversationSort === "created"}
            onClick={() => {
              setConversationSort("created");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={ClockArrowDown}
            label={t(I18nKey.CONVERSATION_PANEL$SORT_UPDATED)}
            selected={conversationSort === "updated"}
            onClick={() => {
              setConversationSort("updated");
              setFilterMenuOpen(false);
            }}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SHOW)}</MenuHeading>
          <MenuRow
            icon={MessageCircle}
            label={t(I18nKey.CONVERSATION_PANEL$ALL_THREADS)}
            selected={threadScope === "all"}
            onClick={() => {
              setThreadScope("all");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Star}
            label={t(I18nKey.CONVERSATION_PANEL$RELEVANT_THREADS)}
            selected={threadScope === "relevant"}
            onClick={() => {
              setThreadScope("relevant");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Archive}
            label={t(I18nKey.CONVERSATION_PANEL$SHOW_ARCHIVED)}
            selected={showArchivedConversations}
            testId="toggle-show-archived"
            onClick={() => {
              toggleShowArchivedConversations();
              setFilterMenuOpen(false);
            }}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS)}</MenuHeading>
          <MenuRow
            icon={MessageCircle}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ALL)}
            selected={automationFilterMode === "all"}
            testId="automation-filter-all"
            onClick={() => {
              setAutomationFilterMode("all");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={EyeOff}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_HIDE)}
            selected={automationFilterMode === "hide-automations"}
            testId="automation-filter-hide"
            onClick={() => {
              setAutomationFilterMode("hide-automations");
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Workflow}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ONLY)}
            selected={automationFilterMode === "only-automations"}
            testId="automation-filter-only"
            onClick={() => {
              setAutomationFilterMode("only-automations");
              setFilterMenuOpen(false);
            }}
          />
          {automationFilterMode === "only-automations"
            ? automationNameFacets.map((facet) => (
                <MenuRow
                  key={facet}
                  icon={Tag}
                  label={
                    facet === UNNAMED_AUTOMATION_FACET
                      ? t(I18nKey.CONVERSATION_PANEL$AUTOMATION_UNNAMED)
                      : facet
                  }
                  selected={selectedAutomationNames.includes(facet)}
                  testId={`automation-name-filter-${facet}`}
                  // Multi-select name rows keep the menu open (unlike the
                  // mode radios above) so several names can be toggled in
                  // one visit.
                  onClick={() => onToggleAutomationName(facet)}
                />
              ))
            : null}

          {tagFacets.length > 0 ? (
            <>
              <MenuSeparator />
              <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$TAGS)}</MenuHeading>
              {tagFacets.map((facet) => (
                <MenuRow
                  key={facet}
                  icon={Tag}
                  label={facet}
                  selected={selectedTagFacets.includes(facet)}
                  testId={`tag-filter-${facet}`}
                  // Multi-select facet rows keep the menu open so several
                  // facets can be toggled in one visit (same behavior as the
                  // automation name rows above).
                  onClick={() => onToggleTagFacet(facet)}
                />
              ))}
            </>
          ) : null}

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$METADATA)}</MenuHeading>
          <MenuRow
            icon={GitBranch}
            label={t(I18nKey.CONVERSATION_PANEL$REPO_BRANCH)}
            selected={showRepoBranchMetadata}
            testId="toggle-repo-branch-metadata"
            onClick={() => {
              toggleShowRepoBranchMetadata();
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Bot}
            label={t(I18nKey.CONVERSATION_PANEL$LLM_MODEL)}
            selected={showLlmProfiles}
            testId="toggle-llm-profiles"
            onClick={() => {
              toggleShowLlmProfiles();
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={Tag}
            label={t(I18nKey.CONVERSATION_PANEL$TAGS)}
            selected={showTagsMetadata}
            testId="toggle-tags-metadata"
            onClick={() => {
              toggleShowTagsMetadata();
              setFilterMenuOpen(false);
            }}
          />
          <MenuRow
            icon={MousePointerClick}
            label={t(I18nKey.CONVERSATION_PANEL$HOVER_METADATA)}
            selected={showHoverMetadata}
            testId="toggle-hover-metadata"
            onClick={() => {
              toggleShowHoverMetadata();
              setFilterMenuOpen(false);
            }}
          />

          <MenuSeparator />
          <MenuHeading
            suffix={
              <span className="shrink-0 text-right text-[10px] font-medium normal-case tracking-normal text-[var(--oh-muted)]/70">
                {t(I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_HOUR)}
              </span>
            }
          >
            {t(I18nKey.CONVERSATION_PANEL$OLDER_SECTION)}
          </MenuHeading>
          <MenuRow
            testId="toggle-older-conversations"
            icon={showOlderConversations ? EyeOff : Eye}
            label={
              showOlderConversations
                ? capitalizeLabel(t(I18nKey.CONVERSATION$HIDE))
                : capitalizeLabel(t(I18nKey.CONVERSATION$SHOW_ALL))
            }
            onClick={() => {
              toggleShowOlderConversations();
              setFilterMenuOpen(false);
            }}
          />

          <MenuSeparator />
          <MenuRow
            testId="delete-all-conversations"
            icon={Trash2}
            label={capitalizeLabel(t(I18nKey.CONVERSATION$DELETE_ALL))}
            disabled={totalConversationsCount === 0}
            onClick={() => {
              if (totalConversationsCount === 0) return;
              onRequestDeleteAll();
              setFilterMenuOpen(false);
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
