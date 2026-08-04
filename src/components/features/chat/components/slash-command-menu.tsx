import React, { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import {
  dropdownInstantColorClassName,
  dropdownMenuListClassName,
} from "#/utils/dropdown-classes";
import { Text } from "#/ui/typography";
import type { SlashCommandItem } from "#/types/slash-command";
import { I18nKey } from "#/i18n/declaration";
import {
  getSkillDescription,
  stripMarkdown,
} from "#/utils/slash-command-description";

export {
  getSkillDescription,
  stripMarkdown,
} from "#/utils/slash-command-description";

interface SlashCommandMenuItemProps {
  item: SlashCommandItem;
  isSelected: boolean;
  onSelect: (item: SlashCommandItem) => void;
  ref?: React.Ref<HTMLButtonElement>;
}

function SlashCommandMenuItem({
  item,
  isSelected,
  onSelect,
  ref,
}: SlashCommandMenuItemProps) {
  const { t } = useTranslation("openhands");
  const description = useMemo(() => {
    if (item.descriptionKey) return t(item.descriptionKey);
    if ("description" in item.skill && item.skill.description) {
      return stripMarkdown(item.skill.description);
    }
    if ("content" in item.skill && item.skill.content) {
      return getSkillDescription(item.skill.content);
    }
    return null;
  }, [item.descriptionKey, item.skill, t]);

  return (
    <button
      role="option"
      aria-selected={isSelected}
      ref={ref}
      type="button"
      className={cn(
        "w-full px-3 py-2.5 text-left",
        dropdownInstantColorClassName,
        isSelected ? "bg-tertiary" : "hover:bg-[var(--oh-surface-raised)]",
      )}
      onMouseDown={(e) => {
        // Use mouseDown instead of click to fire before input blur
        e.preventDefault();
        onSelect(item);
      }}
    >
      <Text className="font-normal">{item.command}</Text>
      {description && (
        <Text className="text-xs text-[var(--oh-muted)] mt-0.5 truncate block">
          {description}
        </Text>
      )}
    </button>
  );
}

interface SlashCommandMenuProps {
  items: SlashCommandItem[];
  selectedIndex: number;
  onSelect: (item: SlashCommandItem) => void;
}

export function SlashCommandMenu({
  items,
  selectedIndex,
  onSelect,
}: SlashCommandMenuProps) {
  const { t } = useTranslation("openhands");
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // Keep refs array in sync with items length
  useEffect(() => {
    itemRefs.current = itemRefs.current.slice(0, items.length);
  }, [items.length]);

  // Scroll selected item into view
  useEffect(() => {
    const selectedItem = itemRefs.current[selectedIndex];
    if (selectedItem) {
      selectedItem.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  if (items.length === 0) return null;

  return (
    <div
      role="listbox"
      aria-label={t(I18nKey.CHAT_INTERFACE$COMMANDS)}
      className="absolute bottom-full left-0 w-full mb-1 bg-[var(--oh-surface)] border border-[var(--oh-border-subtle)] rounded-lg shadow-lg max-h-[300px] overflow-y-auto custom-scrollbar z-50"
      data-testid="slash-command-menu"
    >
      <div className="px-3 py-2 text-xs text-[var(--oh-muted)] border-b border-[var(--oh-border-subtle)]">
        {t(I18nKey.CHAT_INTERFACE$COMMANDS)}
      </div>
      <div className={dropdownMenuListClassName}>
        {items.map((item, index) => (
          <SlashCommandMenuItem
            key={item.command}
            item={item}
            isSelected={index === selectedIndex}
            onSelect={onSelect}
            ref={(el) => {
              itemRefs.current[index] = el;
            }}
          />
        ))}
      </div>
    </div>
  );
}
