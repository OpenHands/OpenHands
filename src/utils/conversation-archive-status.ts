import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";

export function isArchivedSandboxStatus(
  sandboxStatus: SandboxStatus | null | undefined,
): boolean {
  return sandboxStatus === "MISSING";
}

export function isConversationArchived(
  sandboxStatus: SandboxStatus | null | undefined,
  isExplicitlyArchived: boolean,
): boolean {
  return isExplicitlyArchived || isArchivedSandboxStatus(sandboxStatus);
}
