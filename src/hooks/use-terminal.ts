import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import React from "react";
import { Command, useCommandStore } from "#/stores/command-store";
import { parseTerminalOutput } from "#/utils/parse-terminal-output";

/*
  NOTE: Tests for this hook are indirectly covered by the tests for the XTermTerminal component.
  The reason for this is that the hook exposes a ref that requires a DOM element to be rendered.
*/

const PROMPT = "$ ";

const renderCommand = (command: Command, terminal: Terminal) => {
  const { content, type, alreadyDisplayed } = command;

  // Skip entries already echoed by the interactive stdin path
  if (alreadyDisplayed) {
    return;
  }

  const trimmedContent = (content || "").replaceAll("\n", "\r\n").trim();
  // Only write if there's actual content to avoid empty newlines
  if (trimmedContent) {
    if (type === "input") {
      terminal.write(PROMPT);
    }
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
  if (computedStyle.display === "none") {
    return false;
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

  return true;
};

function resolveTerminalForeground(host: HTMLElement): string {
  const probe = host.ownerDocument.createElement("span");
  probe.style.color = "var(--oh-surface-foreground)";
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.pointerEvents = "none";
  host.appendChild(probe);
  const fromVar = getComputedStyle(probe).color;
  probe.remove();
  if (fromVar && fromVar !== "rgba(0, 0, 0, 0)") {
    return fromVar;
  }
  return getComputedStyle(host).color;
}

// Create a persistent reference that survives component unmounts
// This ensures terminal history is preserved when navigating away and back
const persistentLastCommandIndex = { current: 0 };

export type UseTerminalOptions = {
  /** When set, the terminal accepts keyboard input and submits lines here. */
  onSubmitCommand?: (command: string) => Promise<void>;
};

export const useTerminal = (options: UseTerminalOptions = {}) => {
  const { onSubmitCommand } = options;
  const interactive = typeof onSubmitCommand === "function";
  const onSubmitCommandRef = React.useRef(onSubmitCommand);
  onSubmitCommandRef.current = onSubmitCommand;

  const commands = useCommandStore((state) => state.commands);
  const appendInput = useCommandStore((state) => state.appendInput);
  const appendOutput = useCommandStore((state) => state.appendOutput);

  const terminal = React.useRef<Terminal | null>(null);
  const fitAddon = React.useRef<FitAddon | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const lastCommandIndex = persistentLastCommandIndex; // Use the persistent reference
  const isDisposed = React.useRef(false);
  const inputBuffer = React.useRef("");
  const isRunning = React.useRef(false);

  const createTerminal = (host: HTMLDivElement) =>
    new Terminal({
      fontFamily: "Menlo, Monaco, 'Courier New', monospace",
      fontSize: 14,
      scrollback: 10000,
      scrollSensitivity: 1,
      fastScrollSensitivity: 5,
      disableStdin: !interactive,
      cursorBlink: interactive,
      // Canvas fillStyle does not resolve CSS variables; use transparency so
      // the host / panel background shows through (`allowTransparency` required).
      allowTransparency: true,
      theme: {
        background: "rgba(0, 0, 0, 0)",
        foreground: resolveTerminalForeground(host),
      },
    });

  const fitTerminalSafely = React.useCallback(() => {
    if (isDisposed.current) {
      return;
    }
    if (canFitTerminal(terminal.current, fitAddon.current, ref.current)) {
      fitAddon.current!.fit();
    }
  }, []);

  const writePrompt = React.useCallback(() => {
    if (!terminal.current || isDisposed.current) {
      return;
    }
    terminal.current.write(PROMPT);
  }, []);

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) {
        terminal.current.open(ref.current);
        if (!interactive) {
          // Hide cursor for read-only terminal using ANSI escape sequence
          terminal.current.write("\x1b[?25l");
        }
        fitTerminalSafely();
      }
    }
  };

  // Initialize terminal and handle cleanup
  React.useEffect(() => {
    isDisposed.current = false;
    const host = ref.current;
    if (!host) {
      return undefined;
    }

    terminal.current = createTerminal(host);
    fitAddon.current = new FitAddon();

    if (ref.current) {
      initializeTerminal();
      // Render all commands in array
      // This happens when we just switch to Terminal from other tabs
      if (commands.length > 0) {
        for (let i = 0; i < commands.length; i += 1) {
          renderCommand(commands[i], terminal.current);
        }
        lastCommandIndex.current = commands.length;
      }
      if (interactive) {
        writePrompt();
      }
    }

    const dataDisposable = interactive
      ? terminal.current.onData((data) => {
          if (!terminal.current || isDisposed.current || isRunning.current) {
            return;
          }

          // Enter — submit the line
          if (data === "\r") {
            const command = inputBuffer.current;
            inputBuffer.current = "";
            terminal.current.write("\r\n");

            if (!command.trim()) {
              writePrompt();
              return;
            }

            isRunning.current = true;
            appendInput(command, { alreadyDisplayed: true });

            void (async () => {
              try {
                await onSubmitCommandRef.current?.(command);
              } catch (error) {
                const message =
                  error instanceof Error ? error.message : String(error);
                terminal.current?.writeln(message);
                appendOutput(message, { alreadyDisplayed: true });
              } finally {
                isRunning.current = false;
                if (!isDisposed.current) {
                  writePrompt();
                }
              }
            })();
            return;
          }

          // Backspace
          if (data === "\x7f" || data === "\b") {
            if (inputBuffer.current.length > 0) {
              inputBuffer.current = inputBuffer.current.slice(0, -1);
              terminal.current.write("\b \b");
            }
            return;
          }

          // Ctrl+C — cancel the current line
          if (data === "\x03") {
            inputBuffer.current = "";
            terminal.current.write("^C\r\n");
            writePrompt();
            return;
          }

          // Ignore other control characters (arrows, etc.)
          if (data < " ") {
            return;
          }

          inputBuffer.current += data;
          terminal.current.write(data);
        })
      : null;

    return () => {
      isDisposed.current = true;
      dataDisposable?.dispose();
      terminal.current?.dispose();
      lastCommandIndex.current = 0;
      inputBuffer.current = "";
      isRunning.current = false;
    };
    // Interactive mode is fixed for the lifetime of this mount.
  }, [interactive]);

  React.useEffect(() => {
    if (
      terminal.current &&
      commands.length > 0 &&
      lastCommandIndex.current < commands.length
    ) {
      const pending = commands.slice(lastCommandIndex.current);
      const hasVisibleUpdate = pending.some(
        (command) => !command.alreadyDisplayed,
      );

      // Clear the idle prompt so agent I/O does not stack on "$ $ …"
      if (
        interactive &&
        hasVisibleUpdate &&
        !isRunning.current &&
        inputBuffer.current === ""
      ) {
        terminal.current.write("\r\x1b[2K");
      }

      for (let i = lastCommandIndex.current; i < commands.length; i += 1) {
        renderCommand(commands[i], terminal.current);
      }
      lastCommandIndex.current = commands.length;

      if (interactive && hasVisibleUpdate && !isRunning.current) {
        writePrompt();
      }
    }
  }, [commands, interactive, writePrompt]);

  React.useEffect(() => {
    let resizeObserver: ResizeObserver | null = null;

    resizeObserver = new ResizeObserver(() => {
      // Use requestAnimationFrame to debounce resize events and ensure DOM is ready
      requestAnimationFrame(() => {
        fitTerminalSafely();
      });
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
