import React, { useEffect, useRef } from "react";
import { cn } from "#/utils/utils";
import { SlashCommandItem } from "#/hooks/chat/use-slash-command";

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
      className="absolute bottom-full left-0 w-full mb-1 bg-[#1e2028] border border-[#383b45] rounded-lg shadow-lg max-h-[200px] overflow-y-auto custom-scrollbar z-50"
      data-testid="slash-command-menu"
    >
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
            "w-full px-3 py-2 text-left flex items-center gap-3 transition-colors",
            index === selectedIndex
              ? "bg-[#383b45] text-white"
              : "text-[#d0d9fa] hover:bg-[#2a2d37]",
          )}
          onMouseDown={(e) => {
            // Use mouseDown instead of click to fire before input blur
            e.preventDefault();
            onSelect(item);
          }}
        >
          <span className="text-sm font-mono text-[#7b93db]">
            {item.command}
          </span>
          {item.command !== `/${item.skill.name}` && (
            <span className="text-sm text-[#9ca3af] truncate">
              {item.skill.name}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
