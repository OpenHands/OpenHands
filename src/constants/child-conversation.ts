/**
 * Name of the agent-server tool that starts a child conversation on the backend
 * running the parent. The server registers the tool but never offers it to the
 * user (`user_selectable: false`), so a host attaches it: see
 * `hooks/mutation/ensure-agent-host-tools.ts`.
 */
export const SERVER_CHILD_CONVERSATION_TOOL_NAME = "start_child_conversation";
