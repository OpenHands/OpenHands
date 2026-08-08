import React from "react";
import type { RioTermHandle } from "rioterm";
import { Command, useCommandStore } from "#/stores/command-store";
import { parseTerminalOutput } from "#/utils/parse-terminal-output";

/*
  NOTE: Tests for this hook are covered by __tests__/hooks/use-terminal.test.tsx
  with the rioterm module mocked; the hook exposes a ref that requires a DOM
  element to be rendered.
*/

const renderCommand = (
  command: Command,
  write: (data: string) => void,
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
    write(`${parseTerminalOutput(trimmedContent)}\r\n`);
  }
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

export const useTerminal = () => {
  const commands = useCommandStore((state) => state.commands);
  const handle = React.useRef<RioTermHandle | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const lastCommandIndex = persistentLastCommandIndex; // Use the persistent reference

  const renderCommandsFrom = React.useCallback((start: number) => {
    const terminal = handle.current?.terminal;
    if (!terminal) {
      return;
    }
    const all = useCommandStore.getState().commands;
    for (let i = start; i < all.length; i += 1) {
      if (all[i].type === "input") {
        terminal.write("$ ");
      }
      // Don't pass isUserInput=true so previously streamed commands are
      // shown when the terminal (re)mounts and replays the store
      renderCommand(all[i], (data) => terminal.write(data), false);
    }
    lastCommandIndex.current = all.length;
  }, []);

  // Initialize terminal and handle cleanup
  React.useEffect(() => {
    let disposed = false;
    const host = ref.current;
    if (!host) {
      return undefined;
    }

    const init = async () => {
      // Dynamic import keeps the wasm engine out of the initial bundle
      const { open, defaultTheme } = await import("rioterm");
      if (disposed || !ref.current) {
        return;
      }

      const opened = await open(ref.current, {
        // DOM renderer: rows of real text, so the read-only log stays
        // selectable, translatable, and visible to assistive tech
        renderer: "dom",
        autoFocus: false,
        fontFamily: "Menlo, Monaco, 'Courier New', monospace",
        fontSize: 14,
        scrollback: 10000,
        theme: {
          ...defaultTheme,
          background: "transparent",
          foreground: resolveTerminalForeground(host),
        },
      });

      if (disposed) {
        opened.dispose();
        return;
      }
      handle.current = opened;
      // Hide cursor for read-only terminal using ANSI escape sequence
      opened.terminal.write("\x1b[?25l");
      // Render all commands already in the store. This happens when we
      // just switch to Terminal from other tabs
      renderCommandsFrom(0);
    };

    void init();

    return () => {
      disposed = true;
      handle.current?.dispose();
      handle.current = null;
      lastCommandIndex.current = 0;
    };
  }, []);

  React.useEffect(() => {
    if (
      handle.current &&
      commands.length > 0 &&
      lastCommandIndex.current < commands.length
    ) {
      renderCommandsFrom(lastCommandIndex.current);
    }
  }, [commands, renderCommandsFrom]);

  return ref;
};
