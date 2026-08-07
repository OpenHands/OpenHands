import { create } from "zustand";

export type Command = {
  content: string;
  type: "input" | "output";
  /** Already echoed by the interactive terminal; skip when syncing from the store. */
  alreadyDisplayed?: boolean;
};

interface CommandState {
  commands: Command[];
  appendInput: (
    content: string,
    options?: { alreadyDisplayed?: boolean },
  ) => void;
  appendOutput: (
    content: string,
    options?: { alreadyDisplayed?: boolean },
  ) => void;
  clearTerminal: () => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  commands: [],
  appendInput: (content, options) =>
    set((state) => ({
      commands: [
        ...state.commands,
        {
          content,
          type: "input",
          alreadyDisplayed: options?.alreadyDisplayed,
        },
      ],
    })),
  appendOutput: (content, options) =>
    set((state) => ({
      commands: [
        ...state.commands,
        {
          content,
          type: "output",
          alreadyDisplayed: options?.alreadyDisplayed,
        },
      ],
    })),
  clearTerminal: () => set({ commands: [] }),
}));
