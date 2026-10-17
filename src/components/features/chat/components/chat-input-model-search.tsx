import React from "react";
import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface ChatInputModelSearchProps {
  value: string;
  onChange: (value: string) => void;
  firstOptionRef: React.RefObject<HTMLButtonElement | null>;
  focusOnMount: boolean;
}

export function matchesModelSearch(
  query: string,
  ...values: (string | null | undefined)[]
): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  return values.some((value) => value?.toLowerCase().includes(normalizedQuery));
}

// @spec MDS-004 — Keyboard search and selection
export function ChatInputModelSearch({
  value,
  onChange,
  firstOptionRef,
  focusOnMount,
}: ChatInputModelSearchProps) {
  const { t } = useTranslation("openhands");
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (focusOnMount) inputRef.current?.focus();
  }, [focusOnMount]);

  return (
    <li
      role="presentation"
      className="sticky top-0 z-10 shrink-0 bg-tertiary px-2 py-1"
    >
      <div className="flex items-center gap-2">
        <Search className="size-4 shrink-0 text-muted" aria-hidden />
        <input
          ref={inputRef}
          type="search"
          data-testid="chat-input-model-search"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") return;
            event.stopPropagation();
            // Search lives inside the composer form; Enter must never send a message.
            if (event.key === "Enter") event.preventDefault();
            if (event.key === "ArrowDown" && !event.nativeEvent.isComposing) {
              event.preventDefault();
              firstOptionRef.current?.focus();
            }
          }}
          placeholder={t(I18nKey.MODEL$SEARCH_MODELS)}
          aria-label={t(I18nKey.MODEL$SEARCH_MODELS)}
          className="min-w-0 flex-1 border-0 bg-transparent py-1 text-sm text-foreground outline-none placeholder:text-muted [&::-webkit-search-cancel-button]:hidden"
        />
        {value && (
          <button
            type="button"
            aria-label={t(I18nKey.MCP$SEARCH_CLEAR)}
            onClick={() => {
              onChange("");
              inputRef.current?.focus();
            }}
            className="shrink-0 cursor-pointer rounded p-1 text-muted hover:bg-interactive-hover hover:text-foreground"
          >
            <X className="size-3.5" aria-hidden />
          </button>
        )}
      </div>
    </li>
  );
}
