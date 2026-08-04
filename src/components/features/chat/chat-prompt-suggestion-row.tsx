import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { LayoutGrid } from "lucide-react";
import { useTranslation } from "react-i18next";
import { readScrollFadeState } from "#/components/features/markdown/markdown-table-scroll";
import { AUTOMATIONS_TEMPLATES_PATH } from "#/components/features/automations/automations-page.constants";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  CHAT_PROMPT_SUGGESTIONS,
  type ChatPromptSuggestion,
} from "./chat-prompt-suggestion.constants";

const FADE_WIDTH_CLASS = "w-10";

const suggestionChipClassName = (disabled: boolean) =>
  cn(
    "inline-flex shrink-0 items-center gap-2 rounded-lg border",
    "border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-3 py-1.5",
    "text-sm text-content transition-colors",
    disabled
      ? "cursor-not-allowed opacity-50"
      : "cursor-pointer hover:border-[var(--oh-interactive-hover)] hover:bg-[var(--oh-surface)]",
  );

interface ChatPromptSuggestionRowProps {
  onSuggestionClick: (prompt: string) => void;
  disabled?: boolean;
  suggestions?: ChatPromptSuggestion[];
  showViewAllTemplatesLink?: boolean;
}

export function ChatPromptSuggestionRow({
  onSuggestionClick,
  disabled = false,
  suggestions = CHAT_PROMPT_SUGGESTIONS,
  showViewAllTemplatesLink = true,
}: ChatPromptSuggestionRowProps) {
  const { t } = useTranslation("openhands");
  const scrollRef = useRef<HTMLDivElement>(null);
  const [fadeState, setFadeState] = useState({ left: false, right: false });

  const updateFadeState = useCallback(() => {
    const element = scrollRef.current;
    if (!element) {
      return;
    }
    setFadeState(readScrollFadeState(element));
  }, []);

  useLayoutEffect(() => {
    updateFadeState();

    const element = scrollRef.current;
    if (!element) {
      return undefined;
    }

    const resizeObserver = new ResizeObserver(updateFadeState);
    resizeObserver.observe(element);

    return () => resizeObserver.disconnect();
  }, [updateFadeState, suggestions, showViewAllTemplatesLink]);

  return (
    <div className="relative w-full">
      <div
        ref={scrollRef}
        data-testid="chat-prompt-suggestion-row"
        onScroll={updateFadeState}
        className={cn(
          "flex w-full gap-2 overflow-x-auto",
          "[scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden",
        )}
      >
        {suggestions.map((suggestion) => {
          const Icon = suggestion.icon;

          return (
            <button
              key={suggestion.id}
              type="button"
              disabled={disabled}
              data-testid={`chat-prompt-suggestion-${suggestion.id}`}
              onClick={() => onSuggestionClick(t(suggestion.promptKey))}
              className={suggestionChipClassName(disabled)}
            >
              <Icon
                className="size-4 shrink-0 text-[var(--oh-muted)]"
                aria-hidden
              />
              <span className="whitespace-nowrap">
                {t(suggestion.labelKey)}
              </span>
            </button>
          );
        })}
        {showViewAllTemplatesLink ? (
          <NavigationLink
            to={AUTOMATIONS_TEMPLATES_PATH}
            data-testid="chat-prompt-suggestion-view-all-templates"
            className={suggestionChipClassName(false)}
          >
            <LayoutGrid
              className="size-4 shrink-0 text-[var(--oh-muted)]"
              aria-hidden
            />
            <span className="whitespace-nowrap">
              {t(I18nKey.HOME$SUGGESTION_VIEW_ALL_TEMPLATES)}
            </span>
          </NavigationLink>
        ) : null}
      </div>
      <div
        aria-hidden
        data-testid="chat-prompt-suggestion-fade-left"
        data-visible={fadeState.left ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 left-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-r from-base to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.left ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        aria-hidden
        data-testid="chat-prompt-suggestion-fade-right"
        data-visible={fadeState.right ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 right-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-l from-base to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.right ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
