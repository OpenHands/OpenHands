import { create } from "zustand";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isExecuteBashActionEvent,
  isExecuteBashObservationEvent,
} from "#/types/agent-server/type-guards";

export type Command = {
  content: string;
  type: "input" | "output";
};

interface CommandState {
  commands: Command[];
  appendInput: (content: string) => void;
  appendOutput: (content: string) => void;
  hydrateFromEvents: (events: OpenHandsEvent[]) => void;
  clearTerminal: () => void;
}

const commandsFromEvents = (events: OpenHandsEvent[]): Command[] => {
  const commands: Command[] = [];

  for (const event of events) {
    if (isExecuteBashActionEvent(event)) {
      commands.push({ content: event.action.command, type: "input" });
      continue;
    }

    if (isExecuteBashObservationEvent(event)) {
      const content = event.observation.content
        .filter((item) => item.type === "text")
        .map((item) => item.text)
        .join("\n");
      commands.push({ content, type: "output" });
    }
  }

  return commands;
};

export const useCommandStore = create<CommandState>((set) => ({
  commands: [],
  appendInput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "input" }],
    })),
  appendOutput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "output" }],
    })),
  hydrateFromEvents: (events: OpenHandsEvent[]) =>
    set((state) => {
      // The REST page is loaded before the WebSocket starts with
      // `resend_mode=since`. If live events have already populated the store,
      // leave them in place rather than replaying the same history.
      if (state.commands.length > 0) {
        return state;
      }

      const commands = commandsFromEvents(events);
      return commands.length > 0 ? { commands } : state;
    }),
  clearTerminal: () => set({ commands: [] }),
}));
