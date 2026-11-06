import { ConversationClient } from "@openhands/typescript-client/clients";
import type { ACPPermissionDecision } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface AcpPermissionTarget {
  id: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
}

function client(target: AcpPermissionTarget) {
  return new ConversationClient(getAgentServerClientOptions(target));
}

export const AcpPermissionsService = {
  list: (target: AcpPermissionTarget) =>
    client(target).listACPPermissions(target.id),
  decide: (
    target: AcpPermissionTarget,
    requestId: string,
    decision: ACPPermissionDecision,
  ) => client(target).respondToACPPermission(target.id, requestId, decision),
};
