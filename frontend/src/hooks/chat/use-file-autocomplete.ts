import { useState, useEffect, useCallback, useRef } from "react";
import { useParams } from "react-router";
import ConversationService from "#/api/conversation-service/conversation-service.api";

interface AutocompletePosition {
  top: number;
  left: number;
}

interface AutocompleteState {
  isOpen: boolean;
  files: string[];
  filteredFiles: string[];
  selectedIndex: number;
  position: AutocompletePosition;
  showAbove: boolean;
  query: string;
}

/**
 * Hook for managing file autocomplete functionality in chat input
 * Detects '@' trigger, fetches files, handles keyboard navigation
 */
export function useFileAutocomplete(
  inputRef: React.RefObject<HTMLDivElement | null>,
) {
  const params = useParams<{ conversationId: string }>();
  const [state, setState] = useState<AutocompleteState>({
    isOpen: false,
    files: [],
    filteredFiles: [],
    selectedIndex: 0,
    position: { top: 0, left: 0 },
    showAbove: false,
    query: "",
  });

  // Cache files to avoid repeated API calls
  const filesCache = useRef<string[]>([]);
  const hasFetchedFiles = useRef(false);

  /**
   * Fetch files from API (only once per conversation)
   */
  const fetchFiles = useCallback(async () => {
    if (hasFetchedFiles.current || !params.conversationId) {
      return;
    }

    try {
      const response = await ConversationService.getFiles(
        params.conversationId,
        undefined, // path
        true, // recursive - get all files in repo
      );
      filesCache.current = response || [];
      hasFetchedFiles.current = true;
    } catch (error) {
      // Error fetching files, keep cache empty
      filesCache.current = [];
    }
  }, [params.conversationId]);

  /**
   * Filter files based on query string
   */
  const filterFiles = useCallback((query: string, allFiles: string[]) => {
    if (!query) {
      // Show first 10 files when no query
      return allFiles.slice(0, 10);
    }

    const lowerQuery = query.toLowerCase();
    return allFiles
      .filter((file) => file.toLowerCase().includes(lowerQuery))
      .slice(0, 10); // Limit to 10 results
  }, []);

  /**
   * Get caret coordinates for dropdown positioning
   */
  const getCaretCoordinates = useCallback((): AutocompletePosition | null => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    return {
      top: rect.top,
      left: rect.left,
    };
  }, []);

  /**
   * Determine if dropdown should show above or below cursor
   */
  const shouldShowAbove = useCallback(
    (caretTop: number, caretBottom: number) => {
      const spaceBelow = window.innerHeight - caretBottom;
      const spaceAbove = caretTop;

      return spaceBelow < 250 && spaceAbove > spaceBelow;
    },
    [],
  );

  /**
   * Detect @ trigger and extract query
   */
  const detectTrigger = useCallback((): {
    query: string;
    position: AutocompletePosition;
    showAbove: boolean;
  } | null => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;

    const range = selection.getRangeAt(0);
    const textBeforeCursor =
      range.startContainer.textContent?.slice(0, range.startOffset) || "";

    // Match @word pattern (@ preceded by whitespace or start, followed by 0+ non-whitespace chars)
    // Group 1: whitespace or empty (for start), Group 2: query
    const match = textBeforeCursor.match(/(^|\s)@(\S*)$/);

    if (!match) return null;

    // Get caret position
    const coords = getCaretCoordinates();
    if (!coords) return null;

    // Calculate if should show above
    const rect = range.getBoundingClientRect();
    const showAbove = shouldShowAbove(rect.top, rect.bottom);

    return {
      query: match[2], // Query is now in group 2
      position: {
        top: showAbove ? rect.top : rect.bottom,
        left: coords.left,
      },
      showAbove,
    };
  }, [getCaretCoordinates, shouldShowAbove]);

  /**
   * Insert file path at cursor position
   */
  const insertFile = useCallback(
    (filepath: string) => {
      const element = inputRef.current;
      if (!element) return;

      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0) return;

      const range = selection.getRangeAt(0);
      const textNode = range.startContainer;

      if (textNode.nodeType !== Node.TEXT_NODE) return;

      const text = textNode.textContent || "";
      const cursorPos = range.startOffset;

      // Find the @ symbol before cursor (preceded by whitespace or start)
      // Group 1: whitespace or empty (for start), Group 2: query
      const textBefore = text.slice(0, cursorPos);
      const match = textBefore.match(/(^|\s)@(\S*)$/);

      if (!match) return;

      const atSymbolPos = cursorPos - match[0].length;
      const leadingSpace = match[1]; // Preserve whitespace if present

      // Replace @query with @filepath, preserving any leading space
      const newText = `${text.slice(0, atSymbolPos) + leadingSpace}@${
        filepath
      } ${text.slice(cursorPos)}`;

      textNode.textContent = newText;

      // Move cursor after inserted text (after the space)
      // Account for leadingSpace + @ + filepath + trailing space
      const newCursorPos =
        atSymbolPos + leadingSpace.length + filepath.length + 2; // +2 for @ and space
      const newRange = document.createRange();
      newRange.setStart(textNode, newCursorPos);
      newRange.collapse(true);
      selection.removeAllRanges();
      selection.addRange(newRange);

      // Close autocomplete
      setState((prev) => ({ ...prev, isOpen: false }));

      // Trigger input event to update parent component
      const inputEvent = new Event("input", { bubbles: true });
      element.dispatchEvent(inputEvent);
    },
    [inputRef],
  );

  /**
   * Handle input event - detect @ trigger
   */
  const handleInputForAutocomplete = useCallback(async () => {
    const element = inputRef.current;
    if (!element) return;

    const triggerData = detectTrigger();

    if (triggerData) {
      // @ detected - fetch files if needed and show autocomplete
      await fetchFiles();

      const filtered = filterFiles(triggerData.query, filesCache.current);

      setState({
        isOpen: true,
        files: filesCache.current,
        filteredFiles: filtered,
        selectedIndex: 0,
        position: triggerData.position,
        showAbove: triggerData.showAbove, // Use the correctly calculated value
        query: triggerData.query,
      });
    } else {
      // No @ detected - close autocomplete
      setState((prev) => ({ ...prev, isOpen: false }));
    }
  }, [inputRef, detectTrigger, fetchFiles, filterFiles]);

  /**
   * Handle keyboard navigation
   * Returns true if event was handled by autocomplete
   */
  const handleKeyDownForAutocomplete = useCallback(
    (e: React.KeyboardEvent): boolean => {
      if (!state.isOpen) return false;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setState((prev) => ({
            ...prev,
            selectedIndex: Math.min(
              prev.selectedIndex + 1,
              prev.filteredFiles.length - 1,
            ),
          }));
          return true;

        case "ArrowUp":
          e.preventDefault();
          setState((prev) => ({
            ...prev,
            selectedIndex: Math.max(prev.selectedIndex - 1, 0),
          }));
          return true;

        case "Enter":
          if (state.filteredFiles[state.selectedIndex]) {
            e.preventDefault();
            insertFile(state.filteredFiles[state.selectedIndex]);
            return true;
          }
          return false;

        case "Escape":
          e.preventDefault();
          setState((prev) => ({ ...prev, isOpen: false }));
          return true;

        default:
          return false;
      }
    },
    [state.isOpen, state.selectedIndex, state.filteredFiles, insertFile],
  );

  /**
   * Handle file selection from dropdown
   */
  const handleAutocompleteSelect = useCallback(
    (filepath: string) => {
      insertFile(filepath);
    },
    [insertFile],
  );

  /**
   * Close autocomplete
   */
  const handleAutocompleteClose = useCallback(() => {
    setState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  /**
   * Reset cache when conversation changes
   */
  useEffect(() => {
    hasFetchedFiles.current = false;
    filesCache.current = [];
    setState((prev) => ({ ...prev, isOpen: false }));
  }, [params.conversationId]);

  return {
    isAutocompleteOpen: state.isOpen,
    filteredFiles: state.filteredFiles,
    selectedIndex: state.selectedIndex,
    autocompletePosition: state.position,
    showAbove: state.showAbove,
    handleAutocompleteSelect,
    handleAutocompleteClose,
    handleInputForAutocomplete,
    handleKeyDownForAutocomplete,
  };
}
