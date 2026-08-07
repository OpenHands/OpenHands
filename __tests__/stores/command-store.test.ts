import { beforeEach, describe, expect, it } from "vitest";
import {
  resetCommandStore,
  selectActiveConversationTerminals,
  useCommandStore,
} from "#/stores/command-store";

describe("command-store conversation history + tabs", () => {
  beforeEach(() => {
    resetCommandStore();
  });

  it("keeps separate command history per conversation", () => {
    const store = useCommandStore.getState();
    store.setActiveConversation("conv-a");
    store.appendInput("echo a");
    store.appendOutput("a");

    store.setActiveConversation("conv-b");
    store.appendInput("echo b");
    store.appendOutput("b");

    store.setActiveConversation("conv-a");
    expect(useCommandStore.getState().commands).toEqual([
      { content: "echo a", type: "input", alreadyDisplayed: undefined },
      { content: "a", type: "output", alreadyDisplayed: undefined },
    ]);

    store.setActiveConversation("conv-b");
    expect(useCommandStore.getState().commands).toEqual([
      { content: "echo b", type: "input", alreadyDisplayed: undefined },
      { content: "b", type: "output", alreadyDisplayed: undefined },
    ]);
  });

  it("adds and switches terminal tabs within a conversation", () => {
    const store = useCommandStore.getState();
    store.setActiveConversation("conv-a");
    store.appendOutput("tab-1");

    const secondId = store.addTab();
    expect(secondId).toBeTruthy();
    expect(useCommandStore.getState().commands).toEqual([]);

    store.appendOutput("tab-2");
    const terminals = selectActiveConversationTerminals(
      useCommandStore.getState(),
    );
    expect(terminals?.tabs).toHaveLength(2);
    expect(terminals?.activeTabId).toBe(secondId);
    expect(
      terminals?.tabs.find((tab) => tab.id === secondId)?.commands,
    ).toEqual([
      { content: "tab-2", type: "output", alreadyDisplayed: undefined },
    ]);

    const firstId = terminals!.tabs[0].id;
    store.setActiveTab(firstId);
    expect(useCommandStore.getState().commands).toEqual([
      { content: "tab-1", type: "output", alreadyDisplayed: undefined },
    ]);
  });

  it("refuses to close the last tab and restores another when closing active", () => {
    const store = useCommandStore.getState();
    store.setActiveConversation("conv-a");
    const firstId =
      selectActiveConversationTerminals(useCommandStore.getState())!.tabs[0].id;

    store.closeTab(firstId);
    expect(
      selectActiveConversationTerminals(useCommandStore.getState())?.tabs,
    ).toHaveLength(1);

    const secondId = store.addTab()!;
    store.appendOutput("second");
    store.closeTab(secondId);

    const terminals = selectActiveConversationTerminals(
      useCommandStore.getState(),
    );
    expect(terminals?.tabs).toHaveLength(1);
    expect(terminals?.activeTabId).toBe(firstId);
    expect(useCommandStore.getState().commands).toEqual([]);
  });

  it("clearTerminal only clears the active tab commands", () => {
    const store = useCommandStore.getState();
    store.setActiveConversation("conv-a");
    store.appendOutput("keep-me");
    const firstId =
      selectActiveConversationTerminals(useCommandStore.getState())!.tabs[0].id;

    store.addTab();
    store.appendOutput("clear-me");
    store.clearTerminal();
    expect(useCommandStore.getState().commands).toEqual([]);

    store.setActiveTab(firstId);
    expect(useCommandStore.getState().commands).toEqual([
      { content: "keep-me", type: "output", alreadyDisplayed: undefined },
    ]);
  });
});
