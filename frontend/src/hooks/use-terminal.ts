import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import React from "react";
import { Command, useCommandStore } from "#/state/command-store";
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

// Create a persistent reference that survives component unmounts
// This ensures terminal history is preserved when navigating away and back
const persistentLastCommandIndex = { current: 0 };

export const useTerminal = () => {
  const commands = useCommandStore((state) => state.commands);
  const terminal = React.useRef<Terminal | null>(null);
  const fitAddon = React.useRef<FitAddon | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const lastCommandIndex = persistentLastCommandIndex; // Use the persistent reference

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

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) {
        terminal.current.open(ref.current);
        // Hide cursor for read-only terminal using ANSI escape sequence
        terminal.current.write("\x1b[?25l");
        fitAddon.current?.fit();
      }
    }
  };

  // Initialize terminal and handle cleanup
  React.useEffect(() => {
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

    resizeObserver = new ResizeObserver(() => {
      fitAddon.current?.fit();
    });

    if (ref.current) {
      resizeObserver.observe(ref.current);
    }

    return () => {
      resizeObserver?.disconnect();
    };
  }, []);

  return ref;
};
