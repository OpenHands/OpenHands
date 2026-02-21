import React, { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { SlashCommandItem } from "#/hooks/chat/use-slash-command";

interface SlashCommandMenuProps {
  items: SlashCommandItem[];
  selectedIndex: number;
  onSelect: (item: SlashCommandItem) => void;
}

/**
 * Format a skill name into a human-readable label.
 * e.g. "code-search" -> "Code search", "init" -> "Init"
 */
function formatSkillName(name: string): string {
  return name.replace(/[-_]/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function SlashCommandMenu({
  items,
  selectedIndex,
  onSelect,
}: SlashCommandMenuProps) {
  const { t } = useTranslation();
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
      aria-label="Slash commands"
      className="absolute bottom-full left-0 w-full mb-1 bg-[#1e2028] border border-[#383b45] rounded-lg shadow-lg max-h-[300px] overflow-y-auto custom-scrollbar z-50"
      data-testid="slash-command-menu"
    >
      <div className="px-3 py-2 text-xs text-[#9ca3af] border-b border-[#383b45]">
        {t("CHAT_INTERFACE$COMMANDS")}
      </div>
      {items.map((item, index) => (
        <button
          key={item.command}
          role="option"
          aria-selected={index === selectedIndex}
          ref={(el) => {
            itemRefs.current[index] = el;
          }}
          type="button"
          className={cn(
            "w-full px-3 py-2.5 text-left transition-colors",
            index === selectedIndex ? "bg-[#383b45]" : "hover:bg-[#2a2d37]",
          )}
          onMouseDown={(e) => {
            // Use mouseDown instead of click to fire before input blur
            e.preventDefault();
            onSelect(item);
          }}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-white">
              {formatSkillName(item.skill.name)}
            </span>
            <span className="text-sm text-[#9ca3af]">{item.command}</span>
          </div>
          {item.skill.content && (
            <div className="text-xs text-[#9ca3af] mt-0.5">
              {item.skill.content.match(/^[^.!?\n]*[.!?]/)?.[0] ||
                item.skill.content.split("\n")[0]}
            </div>
          )}
        </button>
      ))}
    </div>
  );
}
