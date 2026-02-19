import { useState, useCallback, useEffect, useMemo } from "react";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { Skill } from "#/api/conversation-service/v1-conversation-service.types";
import { Microagent } from "#/api/open-hands.types";

export type SlashCommandSkill = Skill | Microagent;

export interface SlashCommandItem {
  skill: SlashCommandSkill;
  /** The slash command string, e.g. "/random-number" */
  command: string;
}

/**
 * Get the slash command for a skill. Prefers an existing "/" trigger,
 * otherwise derives one from the skill name.
 */
function getSlashCommand(skill: SlashCommandSkill): string {
  const slashTrigger = skill.triggers?.find((t) => t.startsWith("/"));
  if (slashTrigger) return slashTrigger;
  return `/${skill.name}`;
}

/**
 * Hook for managing slash command autocomplete in the chat input.
 * Detects when user types "/" and provides filtered skill suggestions.
 */
export const useSlashCommand = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
) => {
  const { data: skills } = useConversationSkills();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Build slash command items from all skills
  const slashItems = useMemo(() => {
    if (!skills) return [];
    return skills.map((skill) => ({
      skill,
      command: getSlashCommand(skill),
    }));
  }, [skills]);

  // Filter items based on user input after "/"
  const filteredItems = useMemo(() => {
    if (!filterText) return slashItems;
    const lower = filterText.toLowerCase();
    return slashItems.filter(
      (item) =>
        item.command.slice(1).toLowerCase().includes(lower) ||
        item.skill.name.toLowerCase().includes(lower),
    );
  }, [slashItems, filterText]);

  // Reset selected index when the filter text changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [filterText]);

  // Get the slash command text from the input (e.g., "/hel" -> "hel")
  const getSlashText = useCallback((): string | null => {
    const element = chatInputRef.current;
    if (!element) return null;

    const text = element.innerText || "";
    // Only trigger slash menu when "/" is at the start of the input
    const match = text.match(/^\/(\S*)$/);
    if (match) return match[1];
    return null;
  }, [chatInputRef]);

  // Update the menu state based on current input
  const updateSlashMenu = useCallback(() => {
    const slashText = getSlashText();
    if (slashText !== null && slashItems.length > 0) {
      setFilterText(slashText);
      setIsMenuOpen(true);
    } else {
      setIsMenuOpen(false);
      setFilterText("");
    }
  }, [getSlashText, slashItems.length]);

  // Select an item and replace the input text with the command
  const selectItem = useCallback(
    (item: SlashCommandItem) => {
      const element = chatInputRef.current;
      if (!element) return;

      // Replace the input content with the command + a space
      element.textContent = `${item.command} `;

      // Move cursor to end
      const range = document.createRange();
      const selection = window.getSelection();
      range.selectNodeContents(element);
      range.collapse(false);
      selection?.removeAllRanges();
      selection?.addRange(range);

      // Dispatch input event so smartResize and other handlers fire
      element.dispatchEvent(new Event("input", { bubbles: true }));

      setIsMenuOpen(false);
      setFilterText("");
    },
    [chatInputRef],
  );

  // Handle keyboard navigation in the menu
  const handleSlashKeyDown = useCallback(
    (e: React.KeyboardEvent): boolean => {
      if (!isMenuOpen || filteredItems.length === 0) return false;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < filteredItems.length - 1 ? prev + 1 : 0,
          );
          return true;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev > 0 ? prev - 1 : filteredItems.length - 1,
          );
          return true;
        case "Enter":
        case "Tab":
          e.preventDefault();
          selectItem(filteredItems[selectedIndex]);
          return true;
        case "Escape":
          e.preventDefault();
          setIsMenuOpen(false);
          return true;
        default:
          return false;
      }
    },
    [isMenuOpen, filteredItems, selectedIndex, selectItem],
  );

  const closeMenu = useCallback(() => setIsMenuOpen(false), []);

  return {
    isMenuOpen,
    filteredItems,
    selectedIndex,
    updateSlashMenu,
    selectItem,
    handleSlashKeyDown,
    closeMenu,
  };
};
