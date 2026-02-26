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

/** Get the cursor's character offset within a contentEditable element. */
function getCursorOffset(element: HTMLElement): number {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return -1;
  const range = selection.getRangeAt(0);
  const preRange = range.cloneRange();
  preRange.selectNodeContents(element);
  preRange.setEnd(range.startContainer, range.startOffset);
  return preRange.toString().length;
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

  // Track the character range of the current slash word so selectItem can
  // replace only that portion instead of wiping the entire input.
  const slashRangeRef = useRef<{ start: number; end: number } | null>(null);

  // Detect a slash word at the cursor position.
  // Returns the filter text (characters after "/") and the range of the
  // slash word within the full input text, or null if no slash word found.
  const getSlashText = useCallback((): {
    text: string;
    start: number;
    end: number;
  } | null => {
    const element = chatInputRef.current;
    if (!element) return null;

    // Strip trailing newlines that contentEditable can produce, but preserve
    // spaces so "/command " (after selection) won't re-trigger the menu.
    const text = (element.innerText || "").replace(/[\n\r]+$/, "");
    const cursor = getCursorOffset(element);
    if (cursor < 0) return null;

    const textBeforeCursor = text.slice(0, cursor);
    // Match a "/" preceded by whitespace or at position 0, followed by
    // non-whitespace characters, ending right at the cursor.
    const match = textBeforeCursor.match(/(^|\s)(\/\S*)$/);
    if (!match) return null;

    const slashWord = match[2]; // e.g. "/hel"
    const start = textBeforeCursor.length - slashWord.length;
    // The end of the slash word extends past the cursor to include any
    // contiguous non-whitespace characters (covers the case where the
    // cursor sits in the middle of a word).
    const afterCursor = text.slice(cursor);
    const trailing = afterCursor.match(/^\S*/);
    const end = cursor + (trailing ? trailing[0].length : 0);

    return { text: slashWord.slice(1), start, end }; // strip leading "/"
  }, [chatInputRef]);

  // Update the menu state based on current input
  const updateSlashMenu = useCallback(() => {
    const result = getSlashText();
    if (result !== null && slashItems.length > 0) {
      setFilterText(result.text);
      slashRangeRef.current = { start: result.start, end: result.end };
      setIsMenuOpen(true);
    } else {
      setIsMenuOpen(false);
      setFilterText("");
      slashRangeRef.current = null;
    }
  }, [getSlashText, slashItems.length]);

  // Select an item and replace only the slash word with the command
  const selectItem = useCallback(
    (item: SlashCommandItem) => {
      const element = chatInputRef.current;
      if (!element) return;

      const slashRange = slashRangeRef.current;
      const currentText = (element.innerText || "").replace(/[\n\r]+$/, "");
      const replacement = `${item.command} `;

      if (slashRange) {
        // Splice the command into the text, replacing only the slash word
        element.textContent =
          currentText.slice(0, slashRange.start) +
          replacement +
          currentText.slice(slashRange.end);

        // Position cursor right after the inserted command + space
        const cursorPos = slashRange.start + replacement.length;
        const textNode = element.firstChild;
        if (textNode) {
          const range = document.createRange();
          const sel = window.getSelection();
          const offset = Math.min(cursorPos, textNode.textContent!.length);
          range.setStart(textNode, offset);
          range.collapse(true);
          sel?.removeAllRanges();
          sel?.addRange(range);
        }
      } else {
        // Fallback: replace everything (e.g. if range tracking failed)
        element.textContent = replacement;
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(element);
        range.collapse(false);
        sel?.removeAllRanges();
        sel?.addRange(range);
      }

      setIsMenuOpen(false);
      setFilterText("");
      setSelectedIndex(0);
      slashRangeRef.current = null;

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
        // Cursor-movement keys: close the menu to avoid acting on a stale
        // slash-word range, but don't consume the event so the cursor moves.
        case "ArrowLeft":
        case "ArrowRight":
        case "Home":
        case "End":
          setIsMenuOpen(false);
          return false;
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
