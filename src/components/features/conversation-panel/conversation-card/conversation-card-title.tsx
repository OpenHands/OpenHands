import { cn } from "#/utils/utils";

import { HighlightSearchMatch } from "../highlight-search-match";

export type ConversationCardTitleMode = "view" | "edit";

export type ConversationCardTitleProps = {
  titleMode: ConversationCardTitleMode;
  title: string;
  onSave: (title: string) => void;
  isConversationArchived?: boolean;
  searchQuery?: string;
};

export function ConversationCardTitle({
  titleMode,
  title,
  onSave,
  isConversationArchived,
  searchQuery = "",
}: ConversationCardTitleProps) {
  if (titleMode === "edit") {
    return (
      <input
        /* eslint-disable jsx-a11y/no-autofocus */
        autoFocus
        data-testid="conversation-card-title"
        onClick={(event: React.MouseEvent<HTMLInputElement>) => {
          event.preventDefault();
          event.stopPropagation();
        }}
        onBlur={(e) => {
          const trimmed = e.currentTarget?.value?.trim?.() ?? "";
          onSave(trimmed);
        }}
        onKeyUp={(event: React.KeyboardEvent<HTMLInputElement>) => {
          // Ignore Enter key during IME composition (e.g., Chinese, Japanese, Korean input)
          if (event.nativeEvent.isComposing) {
            return;
          }
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
        type="text"
        defaultValue={title}
        className="text-sm leading-6 font-semibold bg-transparent w-full"
      />
    );
  }

  return (
    <p
      data-testid="conversation-card-title"
      className={cn(
        "text-xs leading-6 font-semibold bg-transparent truncate overflow-hidden",
        isConversationArchived && "opacity-60",
      )}
    >
      <HighlightSearchMatch
        text={title}
        query={searchQuery}
        className={cn(isConversationArchived && "opacity-60")}
      />
    </p>
  );
}
