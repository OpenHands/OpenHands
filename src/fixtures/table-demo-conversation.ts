import type { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";

export const TABLE_DEMO_CONVERSATION_ID = "table-demo";

const TABLE_DEMO_USER_MESSAGE =
  "Can you compare several coding agents in a wide table?";

/** Wide GFM table used to exercise horizontal scrolling in the chat feed. */
export const TABLE_DEMO_AGENT_MARKDOWN = [
  "Here is a wide comparison table. Scroll horizontally to read every column:",
  "",
  "| Feature | OpenHands | Claude Code | Codex | Gemini CLI | Cursor | Windsurf | Cline | Aider | Devin | Continue |",
  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  "| CLI | Yes | Yes | Yes | Yes | IDE | IDE | IDE | Terminal | Cloud | IDE |",
  "| MCP support | Yes | Yes | Partial | Partial | Yes | Yes | Yes | No | Yes | Yes |",
  "| Local models | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes |",
  "| Browser tools | Yes | Yes | Yes | Limited | Yes | Yes | Yes | No | Yes | Yes |",
  "| Git integration | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |",
  "| Best for | General coding | Anthropic stack | OpenAI stack | Google stack | IDE users | IDE users | VS Code | Terminal | Autonomous | IDE users |",
].join("\n");

const TABLE_DEMO_BASE_TIME = Date.UTC(2026, 5, 23, 12, 0, 0);

function createTableDemoMessageEvent(
  id: string,
  role: "user" | "assistant",
  text: string,
  offsetMinutes: number,
): MessageEvent {
  return {
    id,
    timestamp: new Date(
      TABLE_DEMO_BASE_TIME + offsetMinutes * 60_000,
    ).toISOString(),
    source: role === "user" ? "user" : "agent",
    llm_message: {
      role,
      content: [{ type: "text", text }],
    },
    activated_microagents: [],
    extended_content: [],
  };
}

export const TABLE_DEMO_EVENTS: OpenHandsEvent[] = [
  createTableDemoMessageEvent(
    "table-demo-user-1",
    "user",
    TABLE_DEMO_USER_MESSAGE,
    0,
  ),
  createTableDemoMessageEvent(
    "table-demo-agent-1",
    "assistant",
    TABLE_DEMO_AGENT_MARKDOWN,
    1,
  ),
];
