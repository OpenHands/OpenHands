import { useCallback, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useIsExperimentalFeatureEnabled } from "#/hooks/query/use-experimental-features";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  CONVERSATION_MENTIONS_FEATURE_KEY,
  conversationMention,
} from "#/components/features/markdown/conversation-mention";

export interface ConversationMentionItem {
  id: string;
  title: string | null;
}

const MAX_ITEMS = 8;
/** Latest conversations the picker searches, newest activity first. */
const SEARCH_LIMIT = 100;

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

const sameId = (a: string, b: string) =>
  a.replace(/-/g, "").toLowerCase() === b.replace(/-/g, "").toLowerCase();

/**
 * The `#` picker of the chat input: typing `#` and part of a title lists the
 * user's conversations, and choosing one inserts its mention,
 * `[Title](conversation:<id>)`. Only while the `conversation_mentions`
 * experimental feature is on; otherwise it never opens.
 */
export const useConversationMentionPicker = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
) => {
  const enabled = useIsExperimentalFeatureEnabled(
    CONVERSATION_MENTIONS_FEATURE_KEY,
  );
  const { backend } = useActiveBackend();
  const { conversationId } = useOptionalConversationId();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  // Fetched only once the user types `#` with the feature on.
  const { data } = useQuery({
    queryKey: ["conversation-mention-picker", backend.id],
    queryFn: () =>
      AgentServerConversationService.searchConversations(SEARCH_LIMIT),
    enabled: enabled && isMenuOpen,
    staleTime: 15_000,
  });
  const [filterText, setFilterText] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const rangeRef = useRef<{ start: number; end: number } | null>(null);

  const items = useMemo<ConversationMentionItem[]>(() => {
    const words = filterText.toLowerCase().split(/\s+/).filter(Boolean);
    return (data?.items ?? [])
      .filter((c) => !conversationId || !sameId(c.id, conversationId))
      .filter((c) => {
        const title = (c.title ?? "").toLowerCase();
        return words.every((word) => title.includes(word));
      })
      .slice(0, MAX_ITEMS)
      .map((c) => ({ id: c.id, title: c.title }));
  }, [data, filterText, conversationId]);

  // Refs so the key handler always reads the latest values.
  const isMenuOpenRef = useRef(isMenuOpen);
  isMenuOpenRef.current = isMenuOpen;
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const selectedIndexRef = useRef(selectedIndex);
  selectedIndexRef.current = selectedIndex;

  const closeMenu = useCallback(() => {
    setIsMenuOpen(false);
    setFilterText("");
    rangeRef.current = null;
  }, []);

  /** Open the menu on a `#word` that ends at the cursor. */
  const updateMentionMenu = useCallback(() => {
    const element = chatInputRef.current;
    if (!enabled || !element) {
      closeMenu();
      return;
    }
    const text = (element.innerText || "").replace(/[\n\r]+$/, "");
    const cursor = getCursorOffset(element);
    const match =
      cursor < 0 ? null : text.slice(0, cursor).match(/(^|\s)#(\S*)$/);
    if (!match) {
      closeMenu();
      return;
    }
    const start = cursor - match[2].length - 1;
    const trailing = text.slice(cursor).match(/^\S*/);
    rangeRef.current = {
      start,
      end: cursor + (trailing ? trailing[0].length : 0),
    };
    setFilterText(match[2]);
    setSelectedIndex(0);
    setIsMenuOpen(true);
  }, [chatInputRef, enabled, closeMenu]);

  /** Replace the `#word` with the conversation's mention. */
  const selectItem = useCallback(
    (item: ConversationMentionItem) => {
      const element = chatInputRef.current;
      const range = rangeRef.current;
      if (!element || !range) return;
      const text = (element.innerText || "").replace(/[\n\r]+$/, "");
      const replacement = `${conversationMention(item.title, item.id)} `;
      element.textContent =
        text.slice(0, range.start) + replacement + text.slice(range.end);

      const node = element.firstChild;
      if (node) {
        const caret = document.createRange();
        caret.setStart(
          node,
          Math.min(range.start + replacement.length, node.textContent!.length),
        );
        caret.collapse(true);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(caret);
      }
      closeMenu();
      setSelectedIndex(0);
      // Let React's onInput run (resize, draft, submit state).
      element.dispatchEvent(new InputEvent("input", { bubbles: true }));
      element.focus();
    },
    [chatInputRef, closeMenu],
  );

  /** Keyboard navigation; true when the key was for the menu. */
  const handleMentionKeyDown = useCallback(
    (e: React.KeyboardEvent): boolean => {
      const current = itemsRef.current;
      if (!isMenuOpenRef.current || current.length === 0) return false;
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => (i < current.length - 1 ? i + 1 : 0));
          return true;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => (i > 0 ? i - 1 : current.length - 1));
          return true;
        case "Enter":
        case "Tab": {
          const item = current[selectedIndexRef.current];
          if (!item) return false;
          e.preventDefault();
          selectItem(item);
          return true;
        }
        case "Escape":
          e.preventDefault();
          closeMenu();
          return true;
        default:
          return false;
      }
    },
    [selectItem, closeMenu],
  );

  return {
    isMenuOpen: isMenuOpen && items.length > 0,
    items,
    selectedIndex,
    updateMentionMenu,
    selectItem,
    handleMentionKeyDown,
    closeMenu,
  };
};
