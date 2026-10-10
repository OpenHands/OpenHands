import { create } from "zustand";

export type Command = {
  content: string;
  type: "input" | "output";
  /**
   * The ACP `tool_call_id` this row mirrors, when it comes from an ACP tool
   * call. The SDK emits several events per id (a started and a terminal one
   * at least), so ACP mirroring appends each row at most once by checking
   * this field — see `handleACPToolCallInTerminal`.
   */
  toolCallId?: string;
};

interface CommandState {
  commands: Command[];
  appendInput: (content: string, toolCallId?: string) => void;
  appendOutput: (content: string, toolCallId?: string) => void;
  clearTerminal: () => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  commands: [],
  appendInput: (content: string, toolCallId?: string) =>
    set((state) => ({
      commands: [
        ...state.commands,
        { content, type: "input", ...(toolCallId ? { toolCallId } : {}) },
      ],
    })),
  appendOutput: (content: string, toolCallId?: string) =>
    set((state) => ({
      commands: [
        ...state.commands,
        { content, type: "output", ...(toolCallId ? { toolCallId } : {}) },
      ],
    })),
  clearTerminal: () => set({ commands: [] }),
}));
