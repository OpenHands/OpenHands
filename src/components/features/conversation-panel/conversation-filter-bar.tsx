import React from "react";
import { useTranslation } from "react-i18next";
import { Bot, ChevronDown, ChevronUp, Tag, X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  UNNAMED_AUTOMATION_FACET,
  formatTagFacetLabel,
} from "./conversation-panel-list-helpers";

export interface ConversationFilterBarProps {
  /** Raw `key=value` facets; a bare tag is stored as `key=`. */
  tagFacets: string[];
  selectedTagFacets: string[];
  onToggleTagFacet: (facet: string) => void;
  /** Automation names; the unnamed bucket is the constant `__unnamed__`. */
  automationFacets: string[];
  selectedAutomationNames: string[];
  onToggleAutomationName: (name: string) => void;
  onClearAll: () => void;
}

/**
 * 2-row cap on the collapsed chip row, expressed as a CSS height. Chips use
 * `leading-4` (16px) + `py-px` (2px) = 18px per row, plus the container's
 * `gap-1` (4px) between rows = ~40px for two rows. We pad to 44px to absorb
 * any sub-pixel rounding without showing a sliver of a third row. The actual
 * overflow check below compares `scrollHeight` against this same constant
 * (kept in JS so both sides agree).
 */
const COLLAPSED_TWO_ROW_HEIGHT_PX = 44;

export function ConversationFilterBar(
  props: ConversationFilterBarProps,
): React.ReactElement | null {
  const {
    tagFacets,
    selectedTagFacets,
    onToggleTagFacet,
    automationFacets,
    selectedAutomationNames,
    onToggleAutomationName,
    onClearAll,
  } = props;

  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = React.useState(false);
  const innerRef = React.useRef<HTMLDivElement>(null);
  const [overflows, setOverflows] = React.useState(false);

  const hasSelection =
    selectedTagFacets.length > 0 || selectedAutomationNames.length > 0;

  // Measure the un-expanded chip row to decide whether the show-more toggle
  // belongs. We measure against the same two-row constant we set in CSS so
  // the visible clip and the JS check never disagree.
  // NOTE: this hook must run before the empty-state early return below —
  // facets going from zero to non-zero must never change the hook count.
  React.useLayoutEffect(() => {
    const el = innerRef.current;
    if (!el) {
      return;
    }
    // Un-expanded: apply the same max-height the collapsed render uses, so
    // scrollHeight reflects the clipped view, not the full content. We can't
    // read scrollHeight usefully while the element is `overflow: visible`.
    el.style.maxHeight = `${COLLAPSED_TWO_ROW_HEIGHT_PX}px`;
    el.style.overflow = "hidden";
    const overflowing = el.scrollHeight > COLLAPSED_TWO_ROW_HEIGHT_PX;
    setOverflows(overflowing);
    // Reset before paint so the next render can apply the real class names
    // (controlled by `expanded`).
    el.style.maxHeight = "";
    el.style.overflow = "";
  }, [tagFacets, automationFacets, expanded]);

  // The bar carries no useful state when there is nothing to show and nothing
  // selected — keep the DOM empty so it doesn't take up vertical space above
  // the conversation list.
  if (
    tagFacets.length === 0 &&
    automationFacets.length === 0 &&
    !hasSelection
  ) {
    return null;
  }

  // Render every chip every render; CSS clips the visible row when collapsed.
  // The unselected chip count includes the selected facet too (so removing
  // the last selected chip still keeps a stable measure loop).
  const tagChips = tagFacets.map((facet) => {
    const selected = selectedTagFacets.includes(facet);
    return (
      <button
        key={`tag:${facet}`}
        type="button"
        data-testid={`filter-chip-tag-${facet}`}
        aria-pressed={selected}
        onClick={() => onToggleTagFacet(facet)}
        className={cn(
          "inline-flex max-w-full shrink-0 items-center gap-0.5 rounded-md",
          "border px-1.5 py-px text-[10px] leading-4 transition-colors",
          selected
            ? "border-[var(--oh-accent)] bg-[var(--oh-accent)]/15 text-[var(--oh-foreground)]"
            : "border-[var(--oh-border-subtle)] bg-transparent text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-[var(--oh-foreground)]",
        )}
      >
        <Tag aria-hidden className="h-3 w-3 shrink-0" />
        <span className="truncate">{formatTagFacetLabel(facet)}</span>
      </button>
    );
  });

  const automationChips = automationFacets.map((name) => {
    const selected = selectedAutomationNames.includes(name);
    const isUnnamedBucket = name === UNNAMED_AUTOMATION_FACET;
    const label = isUnnamedBucket
      ? t(I18nKey.CONVERSATION_PANEL$AUTOMATION_UNNAMED)
      : name;
    return (
      <button
        key={`automation:${name}`}
        type="button"
        data-testid={`filter-chip-automation-${name}`}
        aria-pressed={selected}
        onClick={() => onToggleAutomationName(name)}
        className={cn(
          "inline-flex max-w-full shrink-0 items-center gap-0.5 rounded-md",
          "border px-1.5 py-px text-[10px] leading-4 transition-colors",
          selected
            ? "border-[var(--oh-accent)] bg-[var(--oh-accent)]/15 text-[var(--oh-foreground)]"
            : "border-[var(--oh-border-subtle)] bg-transparent text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-[var(--oh-foreground)]",
        )}
      >
        <Bot aria-hidden className="h-3 w-3 shrink-0" />
        <span className="truncate">{label}</span>
      </button>
    );
  });

  return (
    <div
      data-testid="conversation-filter-bar"
      className="flex w-full min-w-0 items-start gap-2 px-1 py-1"
    >
      <div
        ref={innerRef}
        data-testid="conversation-filter-bar-chips"
        className={cn(
          "flex min-w-0 flex-1 flex-wrap items-center gap-1",
          !expanded && overflows ? "overflow-hidden" : "",
        )}
        style={
          !expanded && overflows
            ? { maxHeight: `${COLLAPSED_TWO_ROW_HEIGHT_PX}px` }
            : undefined
        }
      >
        {tagChips}
        {automationChips}
      </div>

      {/* reserved: bulk-select affordance docks here (future) */}
      <div className="ml-auto" />

      <div className="flex shrink-0 items-center gap-1">
        {hasSelection ? (
          <button
            type="button"
            data-testid="clear-filters-button"
            onClick={onClearAll}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-1.5 py-px",
              "text-[10px] leading-4 transition-colors",
              "border-[var(--oh-border-subtle)] text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-[var(--oh-foreground)]",
            )}
          >
            <X aria-hidden className="h-3 w-3 shrink-0" />
            <span>{t(I18nKey.CONVERSATION_PANEL$CLEAR_FILTERS)}</span>
          </button>
        ) : null}
        {overflows ? (
          <button
            type="button"
            data-testid="toggle-filters-expanded"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
            className={cn(
              "inline-flex items-center gap-0.5 rounded-md border px-1.5 py-px",
              "text-[10px] leading-4 transition-colors",
              "border-[var(--oh-border-subtle)] text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-[var(--oh-foreground)]",
            )}
          >
            {expanded ? (
              <>
                <ChevronUp aria-hidden className="h-3 w-3 shrink-0" />
                <span>{t(I18nKey.CONVERSATION_PANEL$SHOW_FEWER_FILTERS)}</span>
              </>
            ) : (
              <>
                <ChevronDown aria-hidden className="h-3 w-3 shrink-0" />
                <span>{t(I18nKey.CONVERSATION_PANEL$SHOW_MORE_FILTERS)}</span>
              </>
            )}
          </button>
        ) : null}
      </div>
    </div>
  );
}
