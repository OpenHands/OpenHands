import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { ContentSearchHit } from "#/api/conversation-service/content-search.api";

// Base key of the _zero/_one/_other plural family; i18next picks the form
// from `count`. The generated I18nKey enum only contains the suffixed
// members (same convention as event-group-summary's STEP_KIND_KEYS).
const MATCH_COUNT_KEY = "CONVERSATION_PANEL$CONTENT_SEARCH_MATCH_COUNT";

/**
 * Split a snippet on the [ ] markers the API puts around matched words and
 * render the marked parts emphasized. Markers are unbalanced-proof: a stray
 * opening or closing bracket simply renders literally.
 */
function renderSnippetWithMarks(snippet: string): React.ReactNode[] {
  return snippet.split(/(\[[^\]]*\])/g).map((part, i) => {
    if (part.startsWith("[") && part.endsWith("]") && part.length > 2) {
      return (
        <b key={i} className="font-semibold text-accent">
          {part.slice(1, -1)}
        </b>
      );
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

interface ConversationContentSearchResultsProps {
  hits: ContentSearchHit[];
  /** True while the query is in flight or resolving (kept simple + honest). */
  isPending: boolean;
  /** Set when the last query failed; we show a retryable hint. */
  error: boolean;
  capped: boolean;
  total: number;
  onSelect: (hit: ContentSearchHit) => void;
}

/**
 * Result list for conversation content search. Each row shows the matched
 * snippet with the matched words emphasized plus who said it and when, and
 * opens the conversation at the matched message when clicked.
 */
export function ConversationContentSearchResults({
  hits,
  isPending,
  error,
  capped,
  total,
  onSelect,
}: ConversationContentSearchResultsProps) {
  const { t } = useTranslation("openhands");

  if (error) {
    return (
      <div className="px-4 py-3 text-sm text-danger">
        {t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_ERROR)}
      </div>
    );
  }

  if (isPending) {
    return (
      <div
        data-testid="conversation-content-search-loading"
        className="px-4 py-3 text-sm text-tertiary-alt"
      >
        {t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_LOADING)}
      </div>
    );
  }

  if (hits.length === 0) {
    return (
      <div className="px-4 py-3 text-sm text-tertiary-alt">
        {t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_EMPTY)}
      </div>
    );
  }

  return (
    <div
      data-testid="conversation-content-search-results"
      className="flex flex-col gap-1 py-2"
    >
      <div className="px-3 pb-1 text-xs text-tertiary-alt">
        {t(MATCH_COUNT_KEY, { count: total })}
      </div>
      {hits.map((hit) => (
        <button
          key={`${hit.conversation_id}:${hit.event_id}`}
          type="button"
          onClick={() => onSelect(hit)}
          className={cn(
            "cursor-pointer rounded-lg border border-transparent px-3 py-2 text-left",
            "transition-colors hover:border-border hover:bg-base-secondary",
          )}
        >
          <div className="truncate text-sm font-medium">{hit.title}</div>
          <div className="mt-0.5 line-clamp-2 text-xs text-tertiary-alt">
            {renderSnippetWithMarks(hit.snippet)}
          </div>
          <div className="mt-0.5 text-2xs uppercase tracking-wide text-tertiary-alt">
            {hit.source === "title"
              ? t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_MATCH_TITLE)
              : hit.source}
          </div>
        </button>
      ))}
      {capped && (
        <div className="px-3 pb-1 pt-2 text-2xs text-tertiary-alt">
          {t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_CAPPED)}
        </div>
      )}
    </div>
  );
}
