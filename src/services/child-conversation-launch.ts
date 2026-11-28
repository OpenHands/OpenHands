/**
 * Child-conversation launcher.
 *
 * Child conversations used to be started from the browser through a client
 * tool, but that flow no longer exists. Launching now happens on the
 * agent-server: the server registers a `start_child_conversation` tool that the
 * host attaches to the parent's profile (see
 * `src/hooks/mutation/ensure-agent-host-tools.ts`), so there is no browser-side
 * launcher left in the frontend.
 *
 * This module keeps the well-known `child-conversation-launch.ts` path alive
 * for the verify-openhands feature map (`F27.launch-child`) until that map is
 * updated to name the server-side launcher.
 */
export { SERVER_CHILD_CONVERSATION_TOOL_NAME } from "#/constants/child-conversation";
