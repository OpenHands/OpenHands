import { Tag, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { formatTagFacetLabel } from "./conversation-panel-list-helpers";

interface ConversationActiveTagFiltersProps {
  selectedFacets: readonly string[];
  onToggleFacet: (facet: string) => void;
  onClearAll: () => void;
}

/**
 * Always-visible record of which tag filters are narrowing the list.
 *
 * The facet rows live two levels inside the layouts menu, behind a toggle
 * that itself lives in the advanced-options modal. Without this strip a
 * filter left switched on just makes conversations disappear, with nothing
 * on screen to say why or how to get them back.
 *
 * Renders nothing when no filter is active — an empty bar would cost a row
 * of a narrow sidebar to say "no news".
 */
export function ConversationActiveTagFilters({
  selectedFacets,
  onToggleFacet,
  onClearAll,
}: ConversationActiveTagFiltersProps) {
  const { t } = useTranslation("openhands");

  if (selectedFacets.length === 0) {
    return null;
  }

  return (
    <div
      data-testid="conversation-active-tag-filters"
      className="flex min-w-0 items-start gap-1.5 border-b border-[var(--oh-border)] px-4 py-1.5"
    >
      <Tag
        className="mt-1 h-3 w-3 shrink-0 text-[var(--oh-muted)]"
        aria-hidden
      />

      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
        {selectedFacets.map((facet) => (
          <button
            key={facet}
            type="button"
            data-testid={`active-tag-filter-${facet}`}
            onClick={() => onToggleFacet(facet)}
            className="flex min-w-0 max-w-full cursor-pointer items-center gap-1 rounded-full bg-[var(--oh-surface)] px-2 py-0.5 text-[10px] leading-4 text-white hover:bg-white/10"
          >
            <span className="truncate">{formatTagFacetLabel(facet)}</span>
            <X className="h-3 w-3 shrink-0" aria-hidden />
          </button>
        ))}
      </div>

      <button
        type="button"
        data-testid="clear-tag-filters"
        onClick={onClearAll}
        className="shrink-0 cursor-pointer text-[10px] leading-5 text-[var(--oh-muted)] hover:text-white"
      >
        {t(I18nKey.CONVERSATION_PANEL$CLEAR_FILTERS)}
      </button>
    </div>
  );
}
