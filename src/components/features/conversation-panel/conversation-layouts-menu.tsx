import React from "react";
import { useTranslation } from "react-i18next";
import {
  Clock3,
  Folder,
  ListFilter,
  Shrink,
  SlidersHorizontal,
  Star,
  Tag,
  Trash2,
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
  useConversationPanelPreferencesStore,
  type LayoutSettingsSlice,
} from "#/stores/conversation-panel-preferences-store";
import { formatTagFacetLabel } from "./conversation-panel-list-helpers";
import { MenuHeading } from "./menu-heading";
import { MenuSeparator } from "./menu-separator";
import { MenuRow } from "./menu-row";
import { AdvancedConversationOptionsModal } from "./advanced-conversation-options-modal";

interface LayoutPreset {
  id: string;
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  /** Omitted for the backend-dependent first preset (workspace vs repo). */
  labelKey?: I18nKey;
  settings: Partial<LayoutSettingsSlice>;
}

/**
 * Layout presets are partial preference bundles (design inference from the
 * 2026-08-14 PM walkthrough video; the video names the presets but not their
 * exact bundles). A preset is "active" when every field it names matches the
 * current preferences; any manual deviation reads as "Custom".
 */
const LAYOUT_PRESETS: LayoutPreset[] = [
  { id: "by-workspace", icon: Folder, settings: { organizeMode: "grouped" } },
  {
    id: "recent-activity",
    icon: Clock3,
    labelKey: I18nKey.CONVERSATION_PANEL$LAYOUT_RECENT_ACTIVITY,
    settings: {
      organizeMode: "chronological",
      conversationSort: "updated",
      threadScope: "all",
      showOlderConversations: false,
    },
  },
  {
    id: "focused",
    icon: Star,
    labelKey: I18nKey.CONVERSATION_PANEL$LAYOUT_FOCUSED,
    settings: {
      organizeMode: "chronological",
      conversationSort: "updated",
      threadScope: "relevant",
      showOlderConversations: false,
    },
  },
  {
    id: "minimal",
    icon: Shrink,
    labelKey: I18nKey.CONVERSATION_PANEL$LAYOUT_MINIMAL,
    settings: {
      organizeMode: "chronological",
      conversationSort: "updated",
      threadScope: "relevant",
      showOlderConversations: false,
      showRepoBranchMetadata: false,
      showLlmProfiles: false,
      showTagsMetadata: false,
      showHoverMetadata: false,
    },
  },
];

export function getActiveLayoutPreset(
  settings: LayoutSettingsSlice,
): LayoutPreset | null {
  return (
    LAYOUT_PRESETS.find((preset) =>
      Object.entries(preset.settings).every(
        ([key, value]) => settings[key as keyof LayoutSettingsSlice] === value,
      ),
    ) ?? null
  );
}

export interface ConversationLayoutsMenuProps {
  menuOpen: boolean;
  setMenuOpen: (open: boolean) => void;
  menuRef: React.RefObject<HTMLDivElement | null>;
  backendKind: BackendKind;
  /** Distinct user-facing `key=value` facets among the loaded conversations. */
  tagFacets: readonly string[];
  /** Distinct automation names; the advanced modal owns their facet rows. */
  automationNameFacets: readonly string[];
  totalConversationsCount: number;
  onRequestDeleteAll: () => void;
}

export function ConversationLayoutsMenu({
  menuOpen,
  setMenuOpen,
  menuRef,
  backendKind,
  tagFacets,
  automationNameFacets,
  totalConversationsCount,
  onRequestDeleteAll,
}: ConversationLayoutsMenuProps) {
  const { t } = useTranslation("openhands");
  const preferences = useConversationPanelPreferencesStore();
  const [tagSectionExpanded, setTagSectionExpanded] = React.useState(false);
  const [advancedOpen, setAdvancedOpen] = React.useState(false);

  const groupedLabel =
    backendKind === "local"
      ? t(I18nKey.CONVERSATION_PANEL$BY_WORKSPACE)
      : t(I18nKey.CONVERSATION_PANEL$BY_REPOSITORY);

  const activePreset = getActiveLayoutPreset(preferences);

  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const menuContentRef = React.useRef<HTMLDivElement>(null);

  // Mirror the filter menu's focus management: focus into the menu on open,
  // return focus to the trigger on close.
  const wasOpenRef = React.useRef(menuOpen);
  React.useEffect(() => {
    if (menuOpen) {
      const firstItem =
        menuContentRef.current?.querySelector<HTMLButtonElement>(
          '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]',
        );
      firstItem?.focus();
    } else if (wasOpenRef.current) {
      triggerRef.current?.focus();
    }
    wasOpenRef.current = menuOpen;
  }, [menuOpen]);

  const handleMenuKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setMenuOpen(false);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const container = menuContentRef.current;
    if (!container) return;
    const items = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]',
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
        data-testid="conversation-layouts-toggle"
        aria-label={t(I18nKey.CONVERSATION_PANEL$LAYOUTS_HEADING)}
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        onClick={() => setMenuOpen(!menuOpen)}
        className={cn(
          "relative inline-flex h-7 w-7 items-center justify-center rounded-md text-[var(--oh-muted)] hover:text-white hover:bg-[var(--oh-surface-raised)]",
          dropdownInstantColorClassName,
        )}
      >
        <ListFilter
          className="shrink-0"
          size={14}
          strokeWidth={2}
          aria-hidden
        />
      </button>

      {menuOpen ? (
        <div
          ref={menuContentRef}
          role="menu"
          aria-orientation="vertical"
          aria-label={t(I18nKey.CONVERSATION_PANEL$LAYOUTS_HEADING)}
          tabIndex={-1}
          data-testid="conversation-layouts-menu"
          onKeyDown={handleMenuKeyDown}
          className={cn(
            "absolute right-0 top-full z-50 mt-0 w-64 rounded-md border border-[var(--oh-border-subtle)] bg-tertiary px-1 py-1 text-[var(--oh-foreground)] shadow-lg",
            dropdownMenuListClassName,
            dropdownMenuViewportScrollClassName,
          )}
        >
          <MenuHeading>
            {t(I18nKey.CONVERSATION_PANEL$LAYOUTS_HEADING)}
          </MenuHeading>
          {LAYOUT_PRESETS.map((preset) => (
            <MenuRow
              key={preset.id}
              icon={preset.icon}
              label={preset.labelKey ? t(preset.labelKey) : groupedLabel}
              selected={activePreset?.id === preset.id}
              testId={`layout-preset-${preset.id}`}
              onClick={() => {
                preferences.applyLayoutSettings(preset.settings);
                setMenuOpen(false);
              }}
            />
          ))}

          {preferences.tagFiltersEnabled ? (
            <>
              <MenuSeparator />
              <MenuRow
                icon={Tag}
                label={t(I18nKey.CONVERSATION_PANEL$TAG_FILTERS)}
                testId="tag-filters-section"
                onClick={() => setTagSectionExpanded(!tagSectionExpanded)}
              />
              {tagSectionExpanded ? (
                tagFacets.length > 0 ? (
                  tagFacets.map((facet) => (
                    <MenuRow
                      key={facet}
                      icon={Tag}
                      label={formatTagFacetLabel(facet)}
                      selected={preferences.selectedTagFacets.includes(facet)}
                      testId={`tag-facet-row-${facet}`}
                      onClick={() => preferences.toggleTagFacet(facet)}
                    />
                  ))
                ) : (
                  <p
                    data-testid="tag-filters-empty"
                    className="px-2 py-1 text-[11px] text-[var(--oh-muted)]/70"
                  >
                    {t(I18nKey.CONVERSATION_PANEL$NO_VISIBLE_TAGS)}
                  </p>
                )
              ) : null}
            </>
          ) : null}

          <MenuSeparator />
          <MenuRow
            icon={SlidersHorizontal}
            label={
              activePreset
                ? t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)
                : `${t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)} · ${t(
                    I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS_CUSTOM,
                  )}`
            }
            testId="advanced-options-row"
            onClick={() => {
              setMenuOpen(false);
              setAdvancedOpen(true);
            }}
          />

          <MenuSeparator />
          <MenuRow
            testId="delete-all-conversations"
            icon={Trash2}
            label={capitalizeLabel(t(I18nKey.CONVERSATION$DELETE_ALL))}
            disabled={totalConversationsCount === 0}
            destructive
            onClick={() => {
              if (totalConversationsCount === 0) return;
              onRequestDeleteAll();
              setMenuOpen(false);
            }}
          />
        </div>
      ) : null}

      <AdvancedConversationOptionsModal
        open={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        backendKind={backendKind}
        automationNameFacets={automationNameFacets}
      />
    </div>
  );
}

const capitalizeLabel = (label: string) =>
  label.length > 0 ? label.charAt(0).toUpperCase() + label.slice(1) : label;
