import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import React from "react";
import { Command, useCommandStore } from "#/stores/command-store";
import { parseTerminalOutput } from "#/utils/parse-terminal-output";

/*
  NOTE: Tests for this hook are indirectly covered by the tests for the XTermTerminal component.
  The reason for this is that the hook exposes a ref that requires a DOM element to be rendered.
*/

const renderCommand = (
  command: Command,
  terminal: Terminal,
  isUserInput: boolean = false,
) => {
  const { content, type } = command;

  // Skip rendering user input commands that come from the event stream
  // as they've already been displayed in the terminal as the user typed
  if (type === "input" && isUserInput) {
    return;
  }

  const trimmedContent = (content || "").replaceAll("\n", "\r\n").trim();
  // Only write if there's actual content to avoid empty newlines
  if (trimmedContent) {
    terminal.writeln(parseTerminalOutput(trimmedContent));
  }
};

/**
 * Check if the terminal is ready for fit operations.
 * This prevents the "Cannot read properties of undefined (reading 'dimensions')" error
 * that occurs when fit() is called on a terminal that is hidden, disposed, or not fully initialized.
 */
const canFitTerminal = (
  terminalInstance: Terminal | null,
  fitAddonInstance: FitAddon | null,
  containerElement: HTMLDivElement | null,
): boolean => {
  // #region agent log
  console.log('[DEBUG] canFitTerminal entry', {hasTerminal:!!terminalInstance,hasFitAddon:!!fitAddonInstance,hasContainer:!!containerElement,location:'use-terminal.ts:37',hypothesisId:'H1'});
  // #endregion
  // Check terminal and fitAddon exist
  if (!terminalInstance || !fitAddonInstance) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: missing terminal or fitAddon', {location:'use-terminal.ts:44',hypothesisId:'H2'});
    // #endregion
    return false;
  }

  // Check container element exists
  if (!containerElement) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: missing container', {location:'use-terminal.ts:50',hypothesisId:'H2'});
    // #endregion
    return false;
  }

  // Check element is visible (not display: none)
  // When display is none, offsetParent is null (except for fixed/body elements)
  const computedStyle = window.getComputedStyle(containerElement);
  if (computedStyle.display === "none") {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: element is hidden', {display:computedStyle.display,location:'use-terminal.ts:57',hypothesisId:'H1'});
    // #endregion
    return false;
  }

  // Check element has dimensions
  const { clientWidth, clientHeight } = containerElement;
  if (clientWidth === 0 || clientHeight === 0) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: element has zero dimensions', {clientWidth,clientHeight,location:'use-terminal.ts:63',hypothesisId:'H1'});
    // #endregion
    return false;
  }

  // Check terminal has been opened (element property is set after open())
  if (!terminalInstance.element) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: terminal.element is null', {location:'use-terminal.ts:69',hypothesisId:'H2'});
    // #endregion
    return false;
  }

  // Check viewport exists and has dimensions property
  // The error "Cannot read properties of undefined (reading 'dimensions')" occurs
  // when xterm's Viewport.syncScrollArea tries to access viewport.dimensions
  // but viewport is undefined or dimensions is undefined
  const element = terminalInstance.element as any;
  const viewport = element?.viewport;
  
  // #region agent log
  console.log('[DEBUG] canFitTerminal: checking viewport', {
    hasElement:!!element,
    hasViewport:!!viewport,
    viewportType:viewport?.constructor?.name,
    hasDimensions:!!viewport?.dimensions,
    dimensionsType:viewport?.dimensions?.constructor?.name,
    location:'use-terminal.ts:88',
    hypothesisId:'H5'
  });
  // #endregion
  
  if (!viewport) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: viewport is null/undefined', {location:'use-terminal.ts:95',hypothesisId:'H5'});
    // #endregion
    return false;
  }
  
  // Check if viewport has dimensions property and it's not null/undefined
  if (!viewport.dimensions) {
    // #region agent log
    console.log('[DEBUG] canFitTerminal: viewport.dimensions is null/undefined', {
      viewportKeys:Object.keys(viewport),
      location:'use-terminal.ts:102',
      hypothesisId:'H5'
    });
    // #endregion
    return false;
  }

  // #region agent log
  console.log('[DEBUG] canFitTerminal: all checks passed', {
    hasViewport:!!viewport,
    hasDimensions:!!viewport.dimensions,
    location:'use-terminal.ts:110',
    hypothesisId:'H1'
  });
  // #endregion
  return true;
};

// Create a persistent reference that survives component unmounts
// This ensures terminal history is preserved when navigating away and back
const persistentLastCommandIndex = { current: 0 };

export const useTerminal = () => {
  const commands = useCommandStore((state) => state.commands);
  const terminal = React.useRef<Terminal | null>(null);
  const fitAddon = React.useRef<FitAddon | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const lastCommandIndex = persistentLastCommandIndex; // Use the persistent reference
  const isDisposed = React.useRef(false);

  const createTerminal = () =>
    new Terminal({
      fontFamily: "Menlo, Monaco, 'Courier New', monospace",
      fontSize: 14,
      scrollback: 10000,
      scrollSensitivity: 1,
      fastScrollModifier: "alt",
      fastScrollSensitivity: 5,
      allowTransparency: true,
      disableStdin: true, // Make terminal read-only
      theme: {
        background: "transparent",
      },
    });

  const fitTerminalSafely = React.useCallback((retryCount: number = 0) => {
    // #region agent log
    console.log('[DEBUG] fitTerminalSafely called', {isDisposed:isDisposed.current,hasTerminal:!!terminal.current,hasFitAddon:!!fitAddon.current,hasRef:!!ref.current,retryCount,location:'use-terminal.ts:164',hypothesisId:'H1'});
    // #endregion
    if (isDisposed.current) {
      // #region agent log
      console.log('[DEBUG] Terminal is disposed, skipping fit', {location:'use-terminal.ts:168',hypothesisId:'H4'});
      // #endregion
      return;
    }
    const canFit = canFitTerminal(terminal.current, fitAddon.current, ref.current);
    // #region agent log
    console.log('[DEBUG] canFitTerminal result', {canFit,hasTerminal:!!terminal.current,hasFitAddon:!!fitAddon.current,hasRef:!!ref.current,refDisplay:ref.current?window.getComputedStyle(ref.current).display:'N/A',refDimensions:ref.current?{width:ref.current.clientWidth,height:ref.current.clientHeight}:null,hasElement:!!terminal.current?.element,location:'use-terminal.ts:174',hypothesisId:'H1'});
    // #endregion
    if (canFit) {
      // #region agent log
      console.log('[DEBUG] Calling fit()', {location:'use-terminal.ts:180',hypothesisId:'H1'});
      // #endregion
      try {
        // Double-check viewport before calling fit() to prevent the error
        const element = terminal.current?.element as any;
        const viewport = element?.viewport;
        if (!viewport || !viewport.dimensions) {
          // #region agent log
          console.warn('[DEBUG] fit() skipped: viewport check failed', {
            hasViewport:!!viewport,
            hasDimensions:!!viewport?.dimensions,
            location:'use-terminal.ts:188',
            hypothesisId:'H5'
          });
          // #endregion
          // Retry if viewport is not ready yet (max 10 retries with exponential backoff)
          if (retryCount < 10 && ref.current) {
            const delay = Math.min(200 * Math.pow(1.5, retryCount), 2000);
            setTimeout(() => fitTerminalSafely(retryCount + 1), delay);
          }
          return;
        }
        
        fitAddon.current!.fit();
        // #region agent log
        console.log('[DEBUG] fit() completed successfully', {location:'use-terminal.ts:202',hypothesisId:'H1'});
        // #endregion
      } catch (error) {
        // #region agent log
        console.error('[DEBUG] fit() threw error', {
          error:error instanceof Error?error.message:String(error),
          stack:error instanceof Error?error.stack:undefined,
          location:'use-terminal.ts:207',
          hypothesisId:'H1'
        });
        // #endregion
        console.error('Error fitting terminal:', error);
        // Don't re-throw - just log the error to prevent breaking the app
      }
    } else if (retryCount < 10 && ref.current) {
      // Container not ready yet, retry with exponential backoff
      // Only retry if we haven't exceeded max retries and container exists
      // Increased max retries to 10 and max delay to 2000ms to give more time for container to load
      const delay = Math.min(200 * Math.pow(1.5, retryCount), 2000);
      setTimeout(() => fitTerminalSafely(retryCount + 1), delay);
    }
  }, []);

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) {
        terminal.current.open(ref.current);
        // Hide cursor for read-only terminal using ANSI escape sequence
        terminal.current.write("\x1b[?25l");
        // Wait for container to be ready before fitting
        // Use requestAnimationFrame to ensure DOM is ready, then check dimensions
        // Use multiple requestAnimationFrame calls to ensure DOM is fully ready
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (ref.current && ref.current.clientWidth > 0 && ref.current.clientHeight > 0) {
              fitTerminalSafely(0);
            } else {
              // Container not ready, retry with longer delay and more attempts
              setTimeout(() => fitTerminalSafely(0), 200);
            }
          });
        });
      }
    }
  };

  // Initialize terminal and handle cleanup
  React.useEffect(() => {
    isDisposed.current = false;
    terminal.current = createTerminal();
    fitAddon.current = new FitAddon();

    if (ref.current) {
      initializeTerminal();
      // Render all commands in array
      // This happens when we just switch to Terminal from other tabs
      if (commands.length > 0) {
        for (let i = 0; i < commands.length; i += 1) {
          if (commands[i].type === "input") {
            terminal.current.write("$ ");
          }
          // Don't pass isUserInput=true here because we're initializing the terminal
          // and need to show all previous commands
          renderCommand(commands[i], terminal.current, false);
        }
        lastCommandIndex.current = commands.length;
      }
      // Don't show prompt in read-only terminal
    }

    return () => {
      isDisposed.current = true;
      terminal.current?.dispose();
      lastCommandIndex.current = 0;
    };
  }, []);

  React.useEffect(() => {
    if (
      terminal.current &&
      commands.length > 0 &&
      lastCommandIndex.current < commands.length
    ) {
      for (let i = lastCommandIndex.current; i < commands.length; i += 1) {
        if (commands[i].type === "input") {
          terminal.current.write("$ ");
        }
        // Pass true for isUserInput to skip rendering user input commands
        // that have already been displayed as the user typed
        renderCommand(commands[i], terminal.current, false);
      }
      lastCommandIndex.current = commands.length;
    }
  }, [commands]);

  React.useEffect(() => {
    let resizeObserver: ResizeObserver | null = null;

    resizeObserver = new ResizeObserver((entries) => {
      // #region agent log
      console.log('[DEBUG] ResizeObserver callback triggered', {entriesCount:entries.length,entrySizes:entries.map(e=>({width:e.contentRect.width,height:e.contentRect.height})),location:'use-terminal.ts:281',hypothesisId:'H1'});
      // #endregion
      // Check if container has valid dimensions before attempting to fit
      const entry = entries[0];
      if (entry && entry.contentRect.width > 0 && entry.contentRect.height > 0) {
        // Use requestAnimationFrame to debounce resize events and ensure DOM is ready
        requestAnimationFrame(() => {
          fitTerminalSafely(0);
        });
      } else {
        // Container not ready yet, will retry when dimensions are available
        // #region agent log
        console.log('[DEBUG] ResizeObserver: container not ready, skipping fit', {width:entry?.contentRect.width,height:entry?.contentRect.height,location:'use-terminal.ts:290',hypothesisId:'H1'});
        // #endregion
      }
    });

    if (ref.current) {
      // #region agent log
      console.log('[DEBUG] ResizeObserver observing element', {location:'use-terminal.ts:181',hypothesisId:'H1'});
      // #endregion
      resizeObserver.observe(ref.current);
    }

    return () => {
      // #region agent log
      console.log('[DEBUG] ResizeObserver cleanup', {location:'use-terminal.ts:186',hypothesisId:'H4'});
      // #endregion
      resizeObserver?.disconnect();
    };
  }, [fitTerminalSafely]);

  return ref;
};
