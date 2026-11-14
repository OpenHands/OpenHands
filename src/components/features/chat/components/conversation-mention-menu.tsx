import React, { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import {
  dropdownInstantColorClassName,
  dropdownMenuListClassName,
} from "#/utils/dropdown-classes";
import { Text } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { ConversationMentionItem } from "#/hooks/chat/use-conversation-mention-picker";

interface ConversationMentionMenuProps {
  items: ConversationMentionItem[];
  selectedIndex: number;
  onSelect: (item: ConversationMentionItem) => void;
}

/** The `#` picker's list of conversations, above the chat input. */
export function ConversationMentionMenu({
  items,
  selectedIndex,
  onSelect,
}: ConversationMentionMenuProps) {
  const { t } = useTranslation("openhands");
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    itemRefs.current[selectedIndex]?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  if (items.length === 0) return null;

  return (
    <div
      role="listbox"
      aria-label={t(I18nKey.CHAT_INTERFACE$CONVERSATIONS)}
      className="absolute bottom-full left-0 w-full mb-1 bg-surface border border-border-subtle rounded-lg shadow-lg max-h-75 overflow-y-auto custom-scrollbar z-50"
      data-testid="conversation-mention-menu"
    >
      <div className="px-3 py-2 text-xs text-muted border-b border-border-subtle">
        {t(I18nKey.CHAT_INTERFACE$CONVERSATIONS)}
      </div>
      <div className={dropdownMenuListClassName}>
        {items.map((item, index) => (
          <button
            key={item.id}
            role="option"
            aria-selected={index === selectedIndex}
            ref={(el) => {
              itemRefs.current[index] = el;
            }}
            type="button"
            className={cn(
              "w-full px-3 py-2.5 text-left",
              dropdownInstantColorClassName,
              index === selectedIndex
                ? "bg-tertiary"
                : "hover:bg-surface-raised",
            )}
            onMouseDown={(e) => {
              // mouseDown fires before the input loses focus
              e.preventDefault();
              onSelect(item);
            }}
          >
            <Text className="font-normal truncate block">
              {item.title || t(I18nKey.CHAT_INTERFACE$UNTITLED_CONVERSATION)}
            </Text>
          </button>
        ))}
      </div>
    </div>
  );
}
