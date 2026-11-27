import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import React from "react";
import { useColorTheme } from "#/hooks/use-color-theme";
import { getTerminalTheme } from "#/themes/terminal-themes";
import { useEventStore } from "#/stores/use-event-store";
import { Command, getTerminalCommands } from "#/utils/terminal-commands";
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

export const useTerminal = () => {
  const colorTheme = useColorTheme();
  const events = useEventStore((state) => state.events);
  const commands = React.useMemo(() => getTerminalCommands(events), [events]);
  const terminal = React.useRef<Terminal | null>(null);
  const fitAddon = React.useRef<FitAddon | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const renderedCommands = React.useRef<Command[]>([]);
  const isDisposed = React.useRef(false);

  const createTerminal = (host: HTMLDivElement) =>
    new Terminal({
      fontFamily: "Menlo, Monaco, 'Courier New', monospace",
      fontSize: 14,
      scrollback: 10000,
      scrollSensitivity: 1,
      fastScrollSensitivity: 5,
      disableStdin: true, // Make terminal read-only
      // Canvas fillStyle does not resolve CSS variables; use transparency so
      // the host / panel background shows through (`allowTransparency` required).
      allowTransparency: true,
      theme: getTerminalTheme(colorTheme, resolveTerminalForeground(host)),
    });

  const fitTerminalSafely = React.useCallback(() => {
    if (isDisposed.current) {
      return;
    }
    if (canFitTerminal(terminal.current, fitAddon.current, ref.current)) {
      fitAddon.current!.fit();
    }
  }, []);

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) {
        terminal.current.open(ref.current);
        // Hide cursor for read-only terminal using ANSI escape sequence
        terminal.current.write("\x1b[?25l");
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
    }

    return () => {
      isDisposed.current = true;
      terminal.current?.dispose();
      renderedCommands.current = [];
    };
  }, []);

  React.useEffect(() => {
    const host = ref.current;
    if (!terminal.current?.options || !host) return;
    terminal.current.options.theme = getTerminalTheme(
      colorTheme,
      resolveTerminalForeground(host),
    );
  }, [colorTheme]);

  React.useEffect(() => {
    if (!terminal.current) return;
    const previous = renderedCommands.current;
    const isAppend = previous.every(
      (command, index) =>
        commands[index]?.type === command.type &&
        commands[index]?.content === command.content,
    );
    // Older history pages and conversation switches replace the visible prefix.
    if (!isAppend) {
      terminal.current.reset();
      terminal.current.write("\x1b[?25l");
    }
    for (const command of commands.slice(isAppend ? previous.length : 0)) {
      if (command.type === "input") terminal.current.write("$ ");
      renderCommand(command, terminal.current);
    }
    renderedCommands.current = commands;
  }, [commands]);

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

  return { ref, hasOutput: commands.length > 0 };
};
