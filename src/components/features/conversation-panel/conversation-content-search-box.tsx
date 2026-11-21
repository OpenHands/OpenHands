import React from "react";
import { useTranslation } from "react-i18next";
import { Search, X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ConversationContentSearchBoxProps {
  /** Current query text. */
  query: string;
  /** Fires on every keystroke; the owner debounces before hitting the API. */
  onQueryChange: (query: string) => void;
}

/**
 * Search box for the conversation list, local backends only (the owner
 * renders it conditionally). Styled after the MCP page toolbar so both
 * search experiences feel like the same product.
 */
export function ConversationContentSearchBox({
  query,
  onQueryChange,
}: ConversationContentSearchBoxProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="conversation-content-search"
      className="relative flex min-w-0 items-center rounded-lg border border-border bg-base-secondary transition-colors focus-within:border-contrast/40 focus-within:ring-1 focus-within:ring-contrast/20"
    >
      <Search className="ml-3 h-4 w-4 shrink-0 text-tertiary-alt" aria-hidden />
      <input
        type="search"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder={t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_PLACEHOLDER)}
        aria-label={t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_PLACEHOLDER)}
        data-testid="conversation-content-search-input"
        className={cn(
          "min-w-0 flex-1 border-0 bg-transparent px-3 py-2 text-sm outline-none",
          "placeholder:text-tertiary-alt",
          "[&::-webkit-search-cancel-button]:hidden",
        )}
      />
      {query ? (
        <button
          type="button"
          onClick={() => onQueryChange("")}
          aria-label={t(I18nKey.CONVERSATION_PANEL$CONTENT_SEARCH_CLEAR)}
          data-testid="conversation-content-search-clear"
          className="mr-2 cursor-pointer rounded p-1 text-tertiary-alt hover:text-content"
        >
          <X className="h-4 w-4" />
        </button>
      ) : null}
    </div>
  );
}
