import React, { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Text } from "#/ui/typography";
import { SlashCommandItem } from "#/hooks/chat/use-slash-command";

/**
 * Format a skill name into a human-readable label.
 * e.g. "code-search" -> "Code search", "init" -> "Init"
 */
function formatSkillName(name: string): string {
  return name.replace(/[-_]/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/**
 * Extract a short description from skill content.
 * Tries YAML frontmatter "description:" first, then falls back
 * to the first meaningful line after headers and frontmatter.
 */
export function getSkillDescription(content: string): string | null {
  let body = content;

  // Try to extract description from YAML frontmatter
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const descMatch = frontmatterMatch[1].match(/^description:\s*(.+)$/m);
    if (descMatch) {
      let desc = descMatch[1].trim();
      // Strip surrounding quotes from YAML values
      if (
        (desc.startsWith('"') && desc.endsWith('"')) ||
        (desc.startsWith("'") && desc.endsWith("'"))
      ) {
        desc = desc.slice(1, -1);
      }
      return desc;
    }
    // Skip frontmatter for body parsing
    body = content.slice(frontmatterMatch[0].length);
  }

  // Fall back to first meaningful line (skip headers, empty lines, frontmatter delimiters)
  const meaningful = body
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0 && !line.startsWith("#") && line !== "---");

  if (!meaningful) return null;

  // Return first sentence or whole line
  const sentence = meaningful.match(/^[^.!?\n]*[.!?]/);
  return sentence?.[0] || meaningful;
}

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
  const description = useMemo(
    () => (item.skill.content ? getSkillDescription(item.skill.content) : null),
    [item.skill.content],
  );

  return (
    <button
      role="option"
      aria-selected={isSelected}
      ref={ref}
      type="button"
      className={cn(
        "w-full px-3 py-2.5 text-left transition-colors",
        isSelected ? "bg-[#383b45]" : "hover:bg-[#2a2d37]",
      )}
      onMouseDown={(e) => {
        // Use mouseDown instead of click to fire before input blur
        e.preventDefault();
        onSelect(item);
      }}
    >
      <div className="flex items-center gap-2">
        <Text className="font-semibold">
          {formatSkillName(item.skill.name)}
        </Text>
        <Text className="text-[#9ca3af]">{item.command}</Text>
      </div>
      {description && (
        <div className="text-xs text-[#9ca3af] mt-0.5 truncate">
          {description}
        </div>
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
      aria-label={t("CHAT_INTERFACE$COMMANDS")}
      className="absolute bottom-full left-0 w-full mb-1 bg-[#1e2028] border border-[#383b45] rounded-lg shadow-lg max-h-[300px] overflow-y-auto custom-scrollbar z-50"
      data-testid="slash-command-menu"
    >
      <div className="px-3 py-2 text-xs text-[#9ca3af] border-b border-[#383b45]">
        {t("CHAT_INTERFACE$COMMANDS")}
      </div>
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
  );
}
