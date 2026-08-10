import { create } from "zustand";

export type Command = {
  content: string;
  type: "input" | "output";
  /** Already echoed by the interactive terminal; skip when syncing from the store. */
  alreadyDisplayed?: boolean;
};

export type TerminalTab = {
  id: string;
  /** 1-based display number used for "Terminal N" labels. */
  number: number;
  commands: Command[];
};

export type ConversationTerminals = {
  tabs: TerminalTab[];
  activeTabId: string;
  nextTabNumber: number;
};

type AppendOptions = { alreadyDisplayed?: boolean };

interface CommandState {
  byConversation: Record<string, ConversationTerminals>;
  activeConversationId: string | null;
  /**
   * Mirror of the active tab's commands for existing callers/selectors.
   * Always kept in sync by store actions — do not set directly.
   */
  commands: Command[];

  setActiveConversation: (conversationId: string | null) => void;
  appendInput: (
    content: string,
    options?: AppendOptions,
    conversationId?: string | null,
  ) => void;
  appendOutput: (
    content: string,
    options?: AppendOptions,
    conversationId?: string | null,
  ) => void;
  /** Clears commands in the active tab (keeps the tab). */
  clearTerminal: () => void;
  addTab: () => string | null;
  closeTab: (tabId: string) => void;
  setActiveTab: (tabId: string) => void;
}

function createTab(number: number, commands: Command[] = []): TerminalTab {
  return {
    id: `terminal-${number}-${Math.random().toString(36).slice(2, 10)}`,
    number,
    commands,
  };
}

function createConversationTerminals(
  seedCommands: Command[] = [],
): ConversationTerminals {
  const tab = createTab(1, seedCommands);
  return {
    tabs: [tab],
    activeTabId: tab.id,
    nextTabNumber: 2,
  };
}

function activeCommands(
  byConversation: Record<string, ConversationTerminals>,
  activeConversationId: string | null,
): Command[] {
  if (!activeConversationId) return [];
  const conversation = byConversation[activeConversationId];
  if (!conversation) return [];
  return (
    conversation.tabs.find((tab) => tab.id === conversation.activeTabId)
      ?.commands ?? []
  );
}

function withSyncedCommands(
  partial: Omit<CommandState, "commands"> &
    Partial<Pick<CommandState, keyof CommandState>>,
): CommandState {
  return {
    ...(partial as CommandState),
    commands: activeCommands(
      partial.byConversation,
      partial.activeConversationId,
    ),
  };
}

function updateTabCommands(
  state: CommandState,
  conversationId: string,
  updater: (commands: Command[]) => Command[],
): CommandState {
  const conversation =
    state.byConversation[conversationId] ?? createConversationTerminals();
  const tabs = conversation.tabs.map((tab) =>
    tab.id === conversation.activeTabId
      ? { ...tab, commands: updater(tab.commands) }
      : tab,
  );

  return withSyncedCommands({
    ...state,
    byConversation: {
      ...state.byConversation,
      [conversationId]: {
        ...conversation,
        tabs,
      },
    },
  });
}

function updateActiveTabCommands(
  state: CommandState,
  updater: (commands: Command[]) => Command[],
): CommandState {
  const conversationId = state.activeConversationId;
  if (!conversationId) {
    return state;
  }
  return updateTabCommands(state, conversationId, updater);
}

export const useCommandStore = create<CommandState>((set, get) => ({
  byConversation: {},
  activeConversationId: null,
  commands: [],

  setActiveConversation: (conversationId) =>
    set((state) => {
      if (conversationId === state.activeConversationId) {
        // Still ensure the conversation entry exists.
        if (conversationId && !state.byConversation[conversationId]) {
          return withSyncedCommands({
            ...state,
            byConversation: {
              ...state.byConversation,
              [conversationId]: createConversationTerminals(),
            },
            activeConversationId: conversationId,
          });
        }
        return withSyncedCommands({
          ...state,
          activeConversationId: conversationId,
        });
      }

      if (!conversationId) {
        return withSyncedCommands({
          ...state,
          activeConversationId: null,
        });
      }

      const byConversation = state.byConversation[conversationId]
        ? state.byConversation
        : {
            ...state.byConversation,
            [conversationId]: createConversationTerminals(),
          };

      return withSyncedCommands({
        ...state,
        byConversation,
        activeConversationId: conversationId,
      });
    }),

  appendInput: (content, options, conversationId) =>
    set((state) => {
      const targetId = conversationId ?? state.activeConversationId;
      if (!targetId) return state;
      const withActive =
        state.activeConversationId == null
          ? { ...state, activeConversationId: targetId }
          : state;
      return updateTabCommands(withActive, targetId, (commands) => [
        ...commands,
        {
          content,
          type: "input",
          alreadyDisplayed: options?.alreadyDisplayed,
        },
      ]);
    }),

  appendOutput: (content, options, conversationId) =>
    set((state) => {
      const targetId = conversationId ?? state.activeConversationId;
      if (!targetId) return state;
      const withActive =
        state.activeConversationId == null
          ? { ...state, activeConversationId: targetId }
          : state;
      return updateTabCommands(withActive, targetId, (commands) => [
        ...commands,
        {
          content,
          type: "output",
          alreadyDisplayed: options?.alreadyDisplayed,
        },
      ]);
    }),

  clearTerminal: () => set((state) => updateActiveTabCommands(state, () => [])),

  addTab: () => {
    const state = get();
    const conversationId = state.activeConversationId;
    if (!conversationId) return null;

    const conversation =
      state.byConversation[conversationId] ?? createConversationTerminals();
    const tab = createTab(conversation.nextTabNumber);
    set(
      withSyncedCommands({
        ...state,
        byConversation: {
          ...state.byConversation,
          [conversationId]: {
            tabs: [...conversation.tabs, tab],
            activeTabId: tab.id,
            nextTabNumber: conversation.nextTabNumber + 1,
          },
        },
      }),
    );
    return tab.id;
  },

  closeTab: (tabId) =>
    set((state) => {
      const conversationId = state.activeConversationId;
      if (!conversationId) return state;
      const conversation = state.byConversation[conversationId];
      if (!conversation || conversation.tabs.length <= 1) return state;

      const tabs = conversation.tabs.filter((tab) => tab.id !== tabId);
      if (tabs.length === conversation.tabs.length) return state;

      const activeTabId =
        conversation.activeTabId === tabId
          ? tabs[tabs.length - 1].id
          : conversation.activeTabId;

      return withSyncedCommands({
        ...state,
        byConversation: {
          ...state.byConversation,
          [conversationId]: {
            ...conversation,
            tabs,
            activeTabId,
          },
        },
      });
    }),

  setActiveTab: (tabId) =>
    set((state) => {
      const conversationId = state.activeConversationId;
      if (!conversationId) return state;
      const conversation = state.byConversation[conversationId];
      if (!conversation?.tabs.some((tab) => tab.id === tabId)) return state;

      return withSyncedCommands({
        ...state,
        byConversation: {
          ...state.byConversation,
          [conversationId]: {
            ...conversation,
            activeTabId: tabId,
          },
        },
      });
    }),
}));

/** Test helper: reset store and optionally seed the active conversation. */
export function resetCommandStore(
  conversationId: string | null = null,
  commands: Command[] = [],
) {
  if (!conversationId) {
    useCommandStore.setState({
      byConversation: {},
      activeConversationId: null,
      commands: [],
    });
    return;
  }

  const conversation = createConversationTerminals(commands);
  useCommandStore.setState({
    byConversation: { [conversationId]: conversation },
    activeConversationId: conversationId,
    commands,
  });
}

export function selectActiveConversationTerminals(
  state: CommandState,
): ConversationTerminals | null {
  if (!state.activeConversationId) return null;
  return state.byConversation[state.activeConversationId] ?? null;
}
