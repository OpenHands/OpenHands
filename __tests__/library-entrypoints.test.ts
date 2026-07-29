import * as publicApi from "../src/index";
import * as browserApi from "../src/components/browser/index";
import * as conversationApi from "../src/components/conversation/index";
import * as filesApi from "../src/components/files/index";
import * as settingsApi from "../src/components/settings/index";
import * as sidebarApi from "../src/components/sidebar/index";
import * as terminalApi from "../src/components/terminal/index";
import { describe, expect, it } from "vitest";

describe("library public entrypoints", () => {
  it("re-exports the primary library surface from the root entry", () => {
    expect(publicApi.ConversationView).toBeTypeOf("function");
    expect(publicApi.ChatPanel).toBeTypeOf("function");
    expect(publicApi.TerminalPanel).toBeTypeOf("function");
    expect(publicApi.BrowserPanel).toBeTypeOf("function");
    expect(publicApi.FileExplorer).toBeTypeOf("function");
    expect(publicApi.SettingsPanel).toBeTypeOf("function");
    expect(publicApi.LLMSettings).toBeTypeOf("function");
    expect(publicApi.Sidebar).toBeTypeOf("function");
    expect(publicApi.ConversationPanel).toBeTypeOf("function");
    expect(publicApi.AgentServerUIProviders).toBeTypeOf("function");
    expect(publicApi.AgentServerUIRoot).toBeTypeOf("function");
    expect(publicApi.AGENT_SERVER_UI_SCOPE_SELECTOR).toBe(
      "[data-agent-server-ui]",
    );
    expect(publicApi.AGENT_SERVER_UI_DEFAULT_THEME).toBe("dark");
  });

  it("keeps each component-domain barrel importable", () => {
    expect(conversationApi.ConversationView).toBeTypeOf("function");
    expect(conversationApi.ChatPanel).toBeTypeOf("function");
    expect(browserApi.BrowserPanel).toBeTypeOf("function");
    expect(terminalApi.TerminalPanel).toBeTypeOf("function");
    expect(filesApi.FileExplorer).toBeTypeOf("function");
    expect(settingsApi.SettingsPanel).toBeTypeOf("function");
    expect(settingsApi.AppSettings).toBeTypeOf("function");
    expect(settingsApi.LLMSettings).toBeTypeOf("function");
    expect(settingsApi.MCPSettings).toBeTypeOf("function");
    expect(settingsApi.SecretsSettings).toBeTypeOf("function");
    expect(sidebarApi.Sidebar).toBeTypeOf("function");
    expect(sidebarApi.ConversationPanel).toBeTypeOf("function");
  });

  it("no longer exposes the removed AgentServerSettings entry", () => {
    expect(
      (settingsApi as Record<string, unknown>).AgentServerSettings,
    ).toBeUndefined();
  });
});
