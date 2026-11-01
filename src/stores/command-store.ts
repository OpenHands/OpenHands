import { create } from "zustand";

export type Command = {
  content: string;
  type: "input" | "output";
};

interface CommandState {
  commands: Command[];
  seenToolCallInputIds: Set<string>;
  seenToolCallOutputs: Map<string, string>;
  appendInput: (content: string) => void;
  appendOutput: (content: string) => void;
  appendACPExecute: (
    command: string,
    output: string | null,
    toolCallId?: string,
  ) => void;
  clearTerminal: () => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  commands: [],
  seenToolCallInputIds: new Set<string>(),
  seenToolCallOutputs: new Map<string, string>(),
  appendInput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "input" }],
    })),
  appendOutput: (content: string) =>
    set((state) => ({
      commands: [...state.commands, { content, type: "output" }],
    })),
  appendACPExecute: (
    command: string,
    output: string | null,
    toolCallId?: string,
  ) =>
    set((state) => {
      const newCommands = [...state.commands];
      const newSeenInputs = new Set(state.seenToolCallInputIds);
      const newSeenOutputs = new Map(state.seenToolCallOutputs);

      const shouldAppendInput =
        Boolean(command?.trim()) &&
        (!toolCallId || !newSeenInputs.has(toolCallId));

      if (shouldAppendInput) {
        if (toolCallId) {
          newSeenInputs.add(toolCallId);
        }
        newCommands.push({ content: command, type: "input" });
      }

      const trimmedOutput = output?.trim();
      let shouldAppendOutput = false;

      if (trimmedOutput) {
        if (!toolCallId) {
          shouldAppendOutput = true;
          newCommands.push({ content: trimmedOutput, type: "output" });
        } else {
          const previousOutput = newSeenOutputs.get(toolCallId);
          if (previousOutput !== trimmedOutput) {
            shouldAppendOutput = true;
            newSeenOutputs.set(toolCallId, trimmedOutput);
            newCommands.push({ content: trimmedOutput, type: "output" });
          }
        }
      }

      if (!shouldAppendInput && !shouldAppendOutput) {
        return state;
      }

      return {
        commands: newCommands,
        seenToolCallInputIds: newSeenInputs,
        seenToolCallOutputs: newSeenOutputs,
      };
    }),
  clearTerminal: () =>
    set({
      commands: [],
      seenToolCallInputIds: new Set(),
      seenToolCallOutputs: new Map(),
    }),
}));
