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
  // Check terminal and fitAddon exist
  if (!terminalInstance || !fitAddonInstance) {
    return false;
  }

  // Check container element exists
  if (!containerElement) {
    return false;
  }

  // Check element is visible (not display: none)
  // When display is none, offsetParent is null (except for fixed/body elements)
  const computedStyle = window.getComputedStyle(containerElement);
  if (
    computedStyle.display === "none" ||
    computedStyle.visibility === "hidden"
  ) {
    return false;
  }

  // Check offsetParent to ensure element is actually rendered in the DOM
  // offsetParent is null for elements with display: none, or elements not in the document flow
  // For body and fixed elements, offsetParent might be null even when visible, so we check dimensions instead
  if (
    containerElement.offsetParent === null &&
    computedStyle.position !== "fixed" &&
    computedStyle.position !== "absolute" &&
    computedStyle.position !== "relative"
  ) {
    // Check if parent is body or html (which is valid)
    const parent = containerElement.parentElement;
    if (parent && parent.tagName !== "BODY" && parent.tagName !== "HTML") {
      return false;
    }
    // If parent is body/html, continue (valid case)
  }

  // Check element has dimensions
  const { clientWidth, clientHeight } = containerElement;
  if (clientWidth === 0 || clientHeight === 0) {
    return false;
  }

  // Check terminal has been opened (element property is set after open())
  if (!terminalInstance.element) {
    return false;
  }

  // Check viewport exists and has dimensions property
  // The error "Cannot read properties of undefined (reading 'dimensions')" occurs
  // when xterm's Viewport.syncScrollArea tries to access viewport.dimensions
  // but viewport is undefined or dimensions is undefined
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const element = terminalInstance.element as any;
  const viewport = element?.viewport;

  if (!viewport) {
    return false;
  }

  // Check if viewport has dimensions property and it's not null/undefined
  if (!viewport.dimensions) {
    return false;
  }

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
      disableStdin: false, // Allow terminal input for interactive commands
      theme: {
        background: "transparent",
      },
    });

  const fitTerminalSafely = React.useCallback((retryCount: number = 0) => {
    if (isDisposed.current) {
      return;
    }
    const canFit = canFitTerminal(
      terminal.current,
      fitAddon.current,
      ref.current,
    );
    if (canFit) {
      try {
        // Double-check viewport before calling fit() to prevent the error
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const element = terminal.current?.element as any;
        const viewport = element?.viewport;
        if (!viewport?.dimensions) {
          // Retry if viewport is not ready yet (max 10 retries with exponential backoff)
          if (retryCount < 10 && ref.current) {
            const delay = Math.min(200 * 1.5 ** retryCount, 2000);
            setTimeout(() => fitTerminalSafely(retryCount + 1), delay);
          }
          return;
        }

        fitAddon.current!.fit();
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Error fitting terminal:", error);
        // Don't re-throw - just log the error to prevent breaking the app
      }
    } else if (retryCount < 10 && ref.current) {
      // Container not ready yet, retry with exponential backoff
      // Only retry if we haven't exceeded max retries and container exists
      // Increased max retries to 10 and max delay to 2000ms to give more time for container to load
      // Check if container is still in DOM and not hidden before retrying
      const container = ref.current;
      const computedStyle = window.getComputedStyle(container);
      const isHidden =
        computedStyle.display === "none" ||
        computedStyle.visibility === "hidden";

      if (!isHidden && container.offsetParent !== null) {
        // Container is visible, retry with exponential backoff
        const delay = Math.min(200 * 1.5 ** retryCount, 2000);
        setTimeout(() => fitTerminalSafely(retryCount + 1), delay);
      } else if (isHidden) {
        // Container is hidden, wait for it to become visible
        // Use ResizeObserver to detect when container becomes visible
        const resizeObserver = new ResizeObserver((entries) => {
          const entry = entries[0];
          if (
            entry &&
            entry.contentRect.width > 0 &&
            entry.contentRect.height > 0
          ) {
            resizeObserver.disconnect();
            // Wait a bit more for DOM to settle, then retry
            setTimeout(() => fitTerminalSafely(0), 100);
          }
        });
        resizeObserver.observe(container);
        // Disconnect after 5 seconds to avoid memory leak
        setTimeout(() => resizeObserver.disconnect(), 5000);
      }
    }
  }, []);

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) {
        terminal.current.open(ref.current);
        // Show cursor for interactive terminal
        terminal.current.write("\x1b[?25h");
        // Wait for container to be ready before fitting
        // Use requestAnimationFrame to ensure DOM is ready, then check dimensions
        // Use multiple requestAnimationFrame calls to ensure DOM is fully ready
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (
              ref.current &&
              ref.current.clientWidth > 0 &&
              ref.current.clientHeight > 0
            ) {
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
      // Terminal is interactive, commands will be executed
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
      // Check if container has valid dimensions before attempting to fit
      const entry = entries[0];
      if (
        entry &&
        entry.contentRect.width > 0 &&
        entry.contentRect.height > 0
      ) {
        // Use requestAnimationFrame to debounce resize events and ensure DOM is ready
        requestAnimationFrame(() => {
          fitTerminalSafely(0);
        });
      }
    });

    if (ref.current) {
      resizeObserver.observe(ref.current);
    }

    return () => {
      resizeObserver?.disconnect();
    };
  }, [fitTerminalSafely]);

  return ref;
};
