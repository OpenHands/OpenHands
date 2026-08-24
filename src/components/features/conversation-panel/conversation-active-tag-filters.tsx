import { Tag, Workflow, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  formatTagFacetLabel,
  UNNAMED_AUTOMATION_FACET,
} from "./conversation-panel-list-helpers";

interface ConversationActiveTagFiltersProps {
  selectedFacets: readonly string[];
  onToggleFacet: (facet: string) => void;
  /** Automation names narrowing the list in `only-automations` mode. */
  selectedAutomationNames: readonly string[];
  onToggleAutomationName: (name: string) => void;
  onClearAll: () => void;
}

/**
 * Always-visible record of which facet selections are narrowing the list —
 * both families: user tags and automation names.
 *
 * The facet rows live two levels inside a menu (tag facets behind a toggle in
 * the layouts menu, automation names inside the advanced-options modal).
 * Without this strip a filter left switched on just makes conversations
 * disappear, with nothing on screen to say why or how to get them back — and
 * both selections are persisted, so a reload brings the narrowing back with
 * the explanation still buried.
 *
 * The automation *mode* deliberately has no chip here: it is a scope, not a
 * facet selection, and `onClearAll` leaves it alone for the same reason —
 * this strip must not silently switch a surface it doesn't show.
 *
 * Renders nothing when nothing is selected — an empty bar would cost a row
 * of a narrow sidebar to say "no news".
 */
export function ConversationActiveTagFilters({
  selectedFacets,
  onToggleFacet,
  selectedAutomationNames,
  onToggleAutomationName,
  onClearAll,
}: ConversationActiveTagFiltersProps) {
  const { t } = useTranslation("openhands");

  if (selectedFacets.length === 0 && selectedAutomationNames.length === 0) {
    return null;
  }

  const chipClassName =
    "flex min-w-0 max-w-full cursor-pointer items-center gap-1 rounded-full bg-[var(--oh-surface)] px-2 py-0.5 text-[10px] leading-4 text-white hover:bg-white/10";

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
            className={chipClassName}
          >
            <span className="truncate">{formatTagFacetLabel(facet)}</span>
            <X className="h-3 w-3 shrink-0" aria-hidden />
          </button>
        ))}

        {selectedAutomationNames.map((name) => (
          <button
            key={name}
            type="button"
            data-testid={`active-automation-filter-${name}`}
            onClick={() => onToggleAutomationName(name)}
            className={chipClassName}
          >
            <Workflow className="h-3 w-3 shrink-0" aria-hidden />
            <span className="truncate">
              {name === UNNAMED_AUTOMATION_FACET
                ? t(I18nKey.CONVERSATION_PANEL$AUTOMATION_UNNAMED)
                : name}
            </span>
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
