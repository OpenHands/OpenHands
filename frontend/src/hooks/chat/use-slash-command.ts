import { useState, useCallback, useEffect, useMemo, useRef } from "react";
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
 * Hook for managing slash command autocomplete in the chat input.
 * Detects when user types "/" and provides filtered skill suggestions.
 * Only skills with explicit "/" triggers (TaskTrigger) appear in the menu.
 */
export const useSlashCommand = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
) => {
  const { data: skills } = useConversationSkills();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Build slash command items from skills:
  // - Skills with explicit "/" triggers use those triggers
  // - AgentSkills without "/" triggers get a derived "/<name>" command
  const slashItems = useMemo(() => {
    if (!skills) return [];
    const items: SlashCommandItem[] = [];
    skills.forEach((skill) => {
      const triggers = skill.triggers || [];
      const slashTriggers = triggers.filter((t) => t.startsWith("/"));

      if (slashTriggers.length > 0) {
        // Skill has explicit slash triggers
        slashTriggers.forEach((trigger) => {
          items.push({ skill, command: trigger });
        });
      } else if (skill.type === "agentskills") {
        // AgentSkills without slash triggers get a derived command
        items.push({ skill, command: `/${skill.name}` });
      }
    });
    return items;
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

  // Keep refs in sync so handleSlashKeyDown always reads the latest values,
  // avoiding stale closures from React's batched state updates.
  const isMenuOpenRef = useRef(isMenuOpen);
  isMenuOpenRef.current = isMenuOpen;
  const filteredItemsRef = useRef(filteredItems);
  filteredItemsRef.current = filteredItems;
  const selectedIndexRef = useRef(selectedIndex);
  selectedIndexRef.current = selectedIndex;

  // Reset selected index when the filter text changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [filterText]);

  // Get the slash command text from the input (e.g., "/hel" -> "hel")
  const getSlashText = useCallback((): string | null => {
    const element = chatInputRef.current;
    if (!element) return null;

    // Strip trailing newlines that contentEditable can produce, but preserve
    // spaces so "/command " (after selection) won't re-trigger the menu.
    const text = (element.innerText || "").replace(/[\n\r]+$/, "");
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
      const sel = window.getSelection();
      range.selectNodeContents(element);
      range.collapse(false);
      sel?.removeAllRanges();
      sel?.addRange(range);

      setIsMenuOpen(false);
      setFilterText("");
      setSelectedIndex(0);

      // Trigger a native InputEvent so React's onInput fires (for smartResize etc.)
      element.dispatchEvent(new InputEvent("input", { bubbles: true }));

      // Restore focus so keyboard events (Enter to submit) work after selection
      element.focus();
    },
    [chatInputRef],
  );

  // Handle keyboard navigation in the menu.
  // Uses refs to always read the latest state, avoiding stale closures.
  const handleSlashKeyDown = useCallback(
    (e: React.KeyboardEvent): boolean => {
      const items = filteredItemsRef.current;
      if (!isMenuOpenRef.current || items.length === 0) return false;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => (prev < items.length - 1 ? prev + 1 : 0));
          return true;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : items.length - 1));
          return true;
        case "Enter":
        case "Tab": {
          const item = items[selectedIndexRef.current];
          if (!item) return false;
          e.preventDefault();
          selectItem(item);
          return true;
        }
        case "Escape":
          e.preventDefault();
          setIsMenuOpen(false);
          return true;
        default:
          return false;
      }
    },
    [selectItem],
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
