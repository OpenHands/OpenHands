import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { v4 as uuidv4 } from "uuid";
import type { HookEvent } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { MCPServerConfig } from "#/types/mcp-server";
import type { SkillInfo } from "#/types/settings";
import type { SlashCommandItem } from "#/types/slash-command";

interface SlashCommandOutputBase {
  id: string;
  anchorEventId: string | null;
}

export interface SlashCommandExtensions {
  skills: SkillInfo[];
  hooks: HookEvent[];
  mcpServers: MCPServerConfig[];
}

export type SlashCommandOutputEntry =
  | (SlashCommandOutputBase & {
      kind: "help";
      commands: SlashCommandItem[];
    })
  | (SlashCommandOutputBase & {
      kind: "skills";
    } & SlashCommandExtensions);

interface SlashCommandOutputState {
  entriesByConversation: Record<string, SlashCommandOutputEntry[]>;
  showHelp: (
    conversationId: string,
    anchorEventId: string | null,
    commands: SlashCommandItem[],
  ) => void;
  showSkills: (
    conversationId: string,
    anchorEventId: string | null,
    extensions: SlashCommandExtensions,
  ) => void;
  clear: (conversationId: string) => void;
  clearAll: () => void;
}

function appendEntry(
  entriesByConversation: SlashCommandOutputState["entriesByConversation"],
  conversationId: string,
  entry: SlashCommandOutputEntry,
) {
  return {
    ...entriesByConversation,
    [conversationId]: [...(entriesByConversation[conversationId] ?? []), entry],
  };
}

export const useSlashCommandOutputStore = create<SlashCommandOutputState>()(
  devtools(
    (set) => ({
      entriesByConversation: {},
      showHelp: (conversationId, anchorEventId, commands) =>
        set((state) => ({
          entriesByConversation: appendEntry(
            state.entriesByConversation,
            conversationId,
            { id: uuidv4(), anchorEventId, kind: "help", commands },
          ),
        })),
      showSkills: (conversationId, anchorEventId, extensions) =>
        set((state) => ({
          entriesByConversation: appendEntry(
            state.entriesByConversation,
            conversationId,
            {
              id: uuidv4(),
              anchorEventId,
              kind: "skills",
              ...extensions,
            },
          ),
        })),
      clear: (conversationId) =>
        set((state) => {
          const entriesByConversation = { ...state.entriesByConversation };
          delete entriesByConversation[conversationId];
          return { entriesByConversation };
        }),
      clearAll: () => set({ entriesByConversation: {} }),
    }),
    { name: "SlashCommandOutputStore" },
  ),
);
